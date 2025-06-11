import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
import numpy as np
import os
from typing import Dict, List
from torch.optim import AdamW

class MedicalData:
    def __init__(self, conversations, summaries, tokenizer, maxlen=512):
        self.conversations = conversations
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        summary = self.summaries[idx]

        # tokenize conversations
        tok_conv = self.tokenizer(
            conversation,
            max_length=self.maxlen,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # tokenize summarize for use in training
        tok_sum = self.tokenizer(
            summary,
            max_length=self.maxlen,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        # change label of padding token to -100 to omit during loss calculation
        labels = tok_sum['input_ids'].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': tok_conv['input_ids'].squeeze(),
            'attention_mask': tok_conv['attention_mask'].squeeze(),
            'labels': labels
        }

class MTST5:
    def __init__(self, model_name="t5-base", batch_size=16, learning_rate=5e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # initialize model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def load_data(self):
        # import data
        train = pd.read_csv("../data/MTS-Dialog-TrainingSet.csv")
        dev = pd.read_csv("../data/MTS-Dialog-ValidationSet.csv")
        
        # create datasets
        train_dataset = MedicalData(
            conversations=train['dialogue'].tolist(),
            summaries=train['section_text'].tolist(),
            tokenizer=self.tokenizer
        )
        
        dev_dataset = MedicalData(
            conversations=dev['dialogue'].tolist(),
            summaries=dev['section_text'].tolist(),
            tokenizer=self.tokenizer
        )
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size)
        
        return train_loader, dev_loader

    def calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores for generated summaries"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            aggregator.add_scores(scores)
        
        result = aggregator.aggregate()
        
        return {
            'rouge1_f1': result['rouge1'].mid.fmeasure,
            'rouge2_f1': result['rouge2'].mid.fmeasure,
            'rougeL_f1': result['rougeL'].mid.fmeasure
        }

    def evaluate(self, dev_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the dev set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating"):
                # move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # generate summaries
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
                
                # decode predictions and references
                predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # replace -100 with pad_token_id for proper decoding
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = self.tokenizer.pad_token_id
                references = self.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(all_predictions, all_references)
        
        return {
            'loss': total_loss / len(dev_loader),
            **rouge_scores
        }

    def train(self, num_epochs=3):
        """Train the model"""
        train_loader, dev_loader = self.load_data()
        
        # initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=total_steps
        )
        
        # training loop
        best_rouge_l = 0
        half_epoch_steps = len(train_loader) // 2  # evaluate on dev every half epoch
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # training mode -- allows weights to update
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            for step, batch in enumerate(progress_bar):
                # move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # backward pass -- updates parameters via gradient descent
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
                
                # evaluate every half epoch
                if (step + 1) % half_epoch_steps == 0:
                    # switch to evaluation mode for validation
                    # all models weights are frozen
                    self.model.evaluate()
                    print(f"\nMid-epoch evaluation at step {step + 1}/{len(train_loader)}")
                    metrics = self.evaluate(dev_loader)
                    print("Validation Metrics:")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")
                    
                    # save best model based on ROUGE-L
                    if metrics['rougeL_f1'] > best_rouge_l:
                        best_rouge_l = metrics['rougeL_f1']
                        self.save_model(f"best_model_epoch_{epoch+1}_step_{step+1}.pt")
                        print(f"New best model saved with ROUGE-L: {best_rouge_l:.4f}")
                    
                    # return to training mode
                    self.model.train()
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"Training Loss: {avg_train_loss:.4f}")
            
            # end of epoch evaluation
            print(f"\nEnd of epoch {epoch + 1} evaluation")
            metrics = self.evaluate(dev_loader)
            print("Validation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # save best model based on ROUGE-L
            if metrics['rougeL_f1'] > best_rouge_l:
                best_rouge_l = metrics['rougeL_f1']
                self.save_model(f"best_model_epoch_{epoch+1}_final.pt")
                print(f"New best model saved with ROUGE-L: {best_rouge_l:.4f}")

    def save_model(self, filename: str):
        """Save model checkpoint"""
        os.makedirs('../MTS_checkpoints', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer_name': self.model_name,
            'rouge_l_score': getattr(self, 'best_rouge_l', 0)
        }
        filepath = os.path.join('..', 'MTS_checkpoints', filename)
        torch.save(checkpoint, filepath)
        
        # Save tokenizer separately
        tokenizer_dir = os.path.join('..', 'MTS_checkpoints', os.path.splitext(filename)[0] + '_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_dir)
        print(f"Model saved to {filepath}")
        print(f"Tokenizer saved to {tokenizer_dir}")


if __name__ == "__main__":
    # create checkpoints directory
    os.makedirs('../MTS_checkpoints', exist_ok=True)
    
    # initialize and train model
    model = MTST5(model_name="t5-base", batch_size=8)
    model.train(num_epochs=5)