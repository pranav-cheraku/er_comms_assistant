import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import pandas as pd
import random
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List
from tqdm import tqdm

class MedicalDataset(Dataset):
    def __init__(self, conversations: List[str], tokenizer: T5Tokenizer, max_length: int = 512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx: int):
        conversation = self.conversations[idx]

        # add task prefix for T5
        input_text = f"summarize: {conversation}"

        # tokenize inputs
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

# load MTS-dialoge data
def load_test_data(tokenizer: T5Tokenizer, batch_size: int = 8):
    # import data
    test = pd.read_csv("../data/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv")
    test_conversations = test['dialogue'].to_list()
    test_summaries = test['section_text'].to_list()

    random.seed(42)  # for reproducibility
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


# load fine-tuned t5 model
def load_model(checkpoint_path: str):
    """Load the fine-tuned model and tokenizer"""
    # set device to CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # load checkpoint on CPU
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # initialize model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # move model to device
    model = model.to(device)
    
    # load tokenizer from the tokenizer directory
    tokenizer = T5Tokenizer.from_pretrained(os.path.join('..', 'MTS_checkpoints', 'MTS_tokenizer'))
    
    return model, tokenizer, device

# calculate rouge scores for evaluation
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
    
    # calculate averages
    return {
        'rouge1_f1': sum(scores['rouge1']) / len(scores['rouge1']),
        'rouge2_f1': sum(scores['rouge2']) / len(scores['rouge2']),
        'rougeL_f1': sum(scores['rougeL']) / len(scores['rougeL'])
    }

# main entrypoint, includes training loop
def main():
    # load model and tokenizer
    model, tokenizer, device = load_model('../MTS_checkpoints/MTS_best_model.pt')
    
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
    
        all_predictions.extend(predictions)
    
    # calculate ROUGE scores
    all_references = test_subset_summaries
    rouge_scores = calculate_rouge_scores(all_predictions, all_references)
    print("\nTest Set ROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"{metric}: {score:.4f}")
    
    # show 3 random examples
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
