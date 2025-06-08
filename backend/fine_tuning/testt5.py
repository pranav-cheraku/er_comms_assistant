import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import pandas as pd
import random
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
from datasets import load_dataset
from tqdm import tqdm

class MedicalDataset(Dataset):
    def __init__(self, conversations: List[str], tokenizer: T5Tokenizer, max_length: int = 512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]

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

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }

def load_test_data(tokenizer, batch_size=8):
    """Load test set and create a random subset of 200 samples"""
    print("Loading dataset...")
    dataset = load_dataset("AGBonnet/augmented-clinical-notes")
    
    # Get all data from train split
    conversations = dataset["train"]["conversation"]
    summaries = dataset["train"]["summary"]
    
    # Calculate split indices for 80/10/10 split
    total_size = len(conversations)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    

    test_conversations = conversations[train_size+val_size:]
    test_summaries = summaries[train_size+val_size:]
    
    # Select random subset of 200 samples
    random.seed(42)  # For reproducibility
    indices = random.sample(range(len(test_conversations)), 200)
    
    test_subset_conversations = [test_conversations[i] for i in indices]
    test_subset_summaries = [test_summaries[i] for i in indices]
    
    # Create dataset and dataloader
    test_dataset = MedicalDataset(
        conversations=test_subset_conversations,
        tokenizer=tokenizer
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return test_loader, test_subset_conversations, test_subset_summaries

def load_model(checkpoint_path):
    """Load the fine-tuned model and tokenizer"""
    # Set device to CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint on CPU
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Initialize model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Load tokenizer from the tokenizer directory
    tokenizer = T5Tokenizer.from_pretrained(os.path.join('..', 'checkpoints_AGB', 'tokenizer'))
    
    return model, tokenizer, device

def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores for generated summaries"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    # Calculate averages
    return {
        'rouge1_f1': sum(scores['rouge1']) / len(scores['rouge1']),
        'rouge2_f1': sum(scores['rouge2']) / len(scores['rouge2']),
        'rougeL_f1': sum(scores['rougeL']) / len(scores['rougeL'])
    }

def main():
    # Load model and tokenizer
    model, tokenizer, device = load_model('../checkpoints_AGB/best_model.pt')
    
    # Load test data and get random subset
    test_loader, test_subset_conversations, test_subset_summaries = load_test_data(tokenizer)
    
    # Generate summaries and collect references
    all_predictions = []
    
    print("\nGenerating summaries for 200 random test samples...")
    for batch in tqdm(test_loader, desc="Testing", total=len(test_loader)):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Generate summaries
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode predictions
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
        all_predictions.extend(predictions)
    
    # Calculate ROUGE scores
    all_references = test_subset_summaries
    rouge_scores = calculate_rouge_scores(all_predictions, all_references)
    print("\nTest Set ROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"{metric}: {score:.4f}")
    
    # Show 3 random examples
    print("\nExample Summaries:")
    example_indices = random.sample(range(len(test_subset_conversations)), 3)
    for idx in example_indices:
        print("\n" + "="*80)
        print("Conversation:")
        print(test_subset_conversations[idx])
        print("\nGenerated Summary:")
        print(all_predictions[idx])
        print("\nReference Summary:")
        print(all_references[idx])
        print("="*80)

if __name__ == "__main__":
    main()
