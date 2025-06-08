import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from datasets import load_dataset
from rouge_score import rouge_scorer, scoring
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, List, Tuple
import json

class MedicalDataset(Dataset):
    def __init__(self, conversations: List[str], summaries: List[str], tokenizer: T5Tokenizer, max_length: int = 512):
        self.conversations = conversations
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        summary = self.summaries[idx]

        # Add task prefix for T5
        input_text = f"summarize: {conversation}"

        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize targets
        targets = self.tokenizer(
            summary,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Replace padding token ids in labels with -100 (ignored in loss calculation)
        labels = targets['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }

def load_data(tokenizer: T5Tokenizer, batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
    """Load and prepare the dataset"""
    print("Loading dataset...")
    dataset = load_dataset("AGBonnet/augmented-clinical-notes")
    
    # Get all data from train split
    conversations = dataset["train"]["conversation"]
    summaries = dataset["train"]["summary"]
    
    # Calculate split indices for 80/10/10 split
    total_size = len(conversations)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    
    # Split into train, validation, and test
    train_conversations = conversations[:train_size]
    train_summaries = summaries[:train_size]
    
    val_conversations = conversations[train_size:train_size + val_size]
    val_summaries = summaries[train_size:train_size + val_size]

    test_conversations = conversations[train_size+val_size:]
    test_summaries = summaries[train_size+val_size:]

    
    # Create train and validation datasets
    train_dataset = MedicalDataset(train_conversations, train_summaries, tokenizer)
    val_dataset = MedicalDataset(val_conversations, val_summaries, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_epoch(model: T5ForConditionalGeneration, 
                train_loader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                device: torch.device,
                gradient_accumulation_steps: int = 4) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass - T5 handles decoder attention internally
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / gradient_accumulation_steps
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
    
    return total_loss / len(train_loader)

def evaluate(model: T5ForConditionalGeneration, 
            val_loader: DataLoader, 
            tokenizer: T5Tokenizer,
            device: torch.device) -> Dict[str, float]:
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass for loss calculation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            # Generate summaries
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                do_sample=False
            )
            
            # Decode predictions and references
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Convert labels back for decoding (replace -100 with pad_token_id)
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            references = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(all_predictions, all_references)
    
    return {
        'loss': total_loss / len(val_loader),
        **rouge_scores
    }

def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    
    for pred, ref in zip(predictions, references):
        if pred.strip() and ref.strip():  # Skip empty predictions/references
            scores = scorer.score(ref, pred)
            aggregator.add_scores(scores)
    
    result = aggregator.aggregate()
    
    return {
        'rouge1_f1': result['rouge1'].mid.fmeasure,
        'rouge2_f1': result['rouge2'].mid.fmeasure,
        'rougeL_f1': result['rougeL'].mid.fmeasure
    }

def save_checkpoint(model: T5ForConditionalGeneration, 
                   tokenizer: T5Tokenizer, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, float],
                   path: str):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    
    # Save tokenizer in the same directory
    tokenizer_path = os.path.join(os.path.dirname(path), "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and tokenizer
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    # Training parameters
    batch_size = 8  
    num_epochs = 1
    learning_rate = 3e-4  
    warmup_steps = 500
    gradient_accumulation_steps = 4 
    
    # Load data
    train_loader, val_loader = load_data(tokenizer, batch_size)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_rouge = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps
        )
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(model, val_loader, tokenizer, device)
        print("\nValidation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save best model based on ROUGE-L
        if metrics['rougeL_f1'] > best_rouge:
            best_rouge = metrics['rougeL_f1']
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                epoch,
                metrics,
                'checkpoints/best_model.pt'
            )
            print(f"New best model saved! ROUGE-L: {best_rouge:.4f}")
        
        # Save regular checkpoint
        save_checkpoint(
            model,
            tokenizer,
            optimizer,
            epoch,
            metrics,
            f'checkpoints/checkpoint_epoch_{epoch + 1}.pt'
        )

if __name__ == "__main__":
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    main()