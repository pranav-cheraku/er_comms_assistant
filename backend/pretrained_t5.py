import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import pandas as pd
import random
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List
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

        # Add sumarize task prefix for T5
        input_text = f"summarize: {conversation}"

        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # return input id's and attention mask for each item dataloader pulls
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }

def load_test_data(tokenizer, batch_size=8):
    """Load test set and create a random subset of 200 samples"""
    print("Loading dataset...")
    dataset = load_dataset("AGBonnet/augmented-clinical-notes")
    
    # extract conversations and summries from dataset
    conversations = dataset["train"]["conversation"]
    summaries = dataset["train"]["summary"]
    
    # calculate split indices for 80/10/10 split
    total_size = len(conversations)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    
    # isolate test set
    test_conversations = conversations[train_size+val_size:]
    test_summaries = summaries[train_size+val_size:]
    
    # Select random subset of 200 samples
    random.seed(42)  # For reproducibility
    indices = random.sample(range(len(test_conversations)), 200)
    
    test_subset_conversations = [test_conversations[i] for i in indices]
    test_subset_summaries = [test_summaries[i] for i in indices]
    
    # create dataset and dataloader
    test_dataset = MedicalDataset(
        conversations=test_subset_conversations,
        tokenizer=tokenizer
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return test_loader, test_subset_conversations, test_subset_summaries


def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores for generated summaries"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    # calculate rouge score for each pair of predictions and ground-truth labels
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    # calculate averages for all three types of rouges scores
    return {
        'rouge1_f1': sum(scores['rouge1']) / len(scores['rouge1']),
        'rouge2_f1': sum(scores['rouge2']) / len(scores['rouge2']),
        'rougeL_f1': sum(scores['rougeL']) / len(scores['rougeL'])
    }

def main():
    # load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load test data and get random subset
    test_loader, test_subset_conversations, test_subset_summaries = load_test_data(tokenizer)
    
    # generate summaries and collect references
    all_predictions = []
    
    print("\nGenerating summaries for 200 random test samples...")
    for batch in tqdm(test_loader, desc="Testing", total=len(test_loader)):
        # move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # generate summaries
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # decode predictions
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # add current predicted sequence to final master list of predictions
        all_predictions.extend(predictions)
    
    # calculate ROUGE scores
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
