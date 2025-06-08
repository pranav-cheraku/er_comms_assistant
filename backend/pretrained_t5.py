import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from rouge_score import rouge_scorer, scoring
import numpy as np
from tqdm import tqdm

print("Importing dataset...")
ds = load_dataset("AGBonnet/augmented-clinical-notes")

# testing t5 out of the box

def import_baseline_t5():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

def summarize_baseline_t5(text, tokenizer, model):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    summary_ids = model.generate(
        input_ids,
        max_length=200,         
        min_length=50,           
        num_beams=5,             
        length_penalty=1.2,      
        early_stopping=True,
        no_repeat_ngram_size=3,  
        temperature=0.7,         
        top_k=50,             
        top_p=0.95,             
        do_sample=True,         
        repetition_penalty=1.2   
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# prompt engineering
def summarize_PE_t5(text, tokenizer, model):
    prompt = """
    You are a medical assistant. You are given a conversation between a doctor and a patient.
    Summarize the following conversation in your own words:

    IMPORTANT:
    Return your summary in the following format. Fill empty fields with the word 'None'.
    {
        "visit motivation": "",
        "admission": [
        {
        "reason": "",
        "date": "",
        "duration": "",
        "care center details": ""
        }
        ],
        "patient information": {
        "age": "",
        "sex": "",
        "ethnicity": "",
        "weight": "",
        "height": "",
        "family medical history": "",
        "recent travels": "",
        "socio economic context": "",
        "occupation": ""
        },
        "patient medical history": {
        "physiological context": "",
        "psychological context": "",
        "vaccination history": "",
        "allergies": "",
        "exercise frequency": "",
        "nutrition": "",
        "sexual history": "",
        "alcohol consumption": "",
        "drug usage": "",
        "smoking status": ""
        },
        "surgeries": [
        {
        "reason": "",
        "Type": "",
        "time": "",
        "outcome": "",
        "details": ""
        }
        ],
        "symptoms": [
        {
        "name of symptom": "",
        "intensity of symptom": "",
        "location": "",
        "time": "",
        "temporalisation": "",
        "behaviours affecting the symptom": "",
        "details": ""
        }
        ],
        "medical examinations": [
        {
        "name": "",
        "result": "",
        "details": ""
        },
        {
        "name": "",
        "result": "",
        "details": ""
        }
        ],
        "diagnosis tests": [
        {
        "test": "",
        "severity": "",
        "result": "",
        "condition": "",
        "time": "",
        "details": ""
        }
        ],
        "treatments": [
        {
        "name": "",
        "related condition": "",
        "dosage": "",
        "time": "",
        "frequency": "",
        "duration": "",
        "reason for taking": "",
        "reaction to treatment": "",
        "details": ""
        }
        ],
        "discharge": {
        "reason": "",
        "referral": "",
        "follow up": "",
        "discharge summary": ""
        }
    }
    """
    prompt += text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    summary_ids = model.generate(
        input_ids,
        max_length=500,          # Increased for structured output
        min_length=100,          # Ensure sufficient detail
        num_beams=5,             # Increased beam size
        length_penalty=1.5,      # Favor longer, more detailed outputs
        early_stopping=True,
        no_repeat_ngram_size=3,  # Prevent repetition
        temperature=0.6,         # Lower temperature for more focused output
        top_k=40,               # More focused token selection
        top_p=0.9,              # Nucleus sampling
        do_sample=True,         # Enable sampling
        repetition_penalty=1.3,  # Stronger repetition penalty
        num_return_sequences=1   # Return only the best sequence
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def calculate_corpus_rouge(predictions, references):
    """Calculate corpus-level ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)  # reference first, prediction second
        aggregator.add_scores(scores)

    result = aggregator.aggregate()

    # Extract precision, recall, f1 for each metric
    rouge_scores = {}
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        # Access the score object's low, mid, and high values
        rouge_scores[f'{metric}_precision'] = result[metric].mid.precision
        rouge_scores[f'{metric}_recall'] = result[metric].mid.recall
        rouge_scores[f'{metric}_f1'] = result[metric].mid.fmeasure

    return rouge_scores


def evaluate_baseline_t5():
    print("Loading T5 model and tokenizer...")
    tokenizer, model = import_baseline_t5()
    
    # Get a subset of the dataset for evaluation
    eval_size = 500  # Adjust this number based on your needs
    permutation = np.random.permutation(len(ds["train"]))
    conversations = [ds["train"]["conversation"][i] for i in permutation[:eval_size]]
    ground_truth_summaries = [ds["train"]["summary"][i] for i in permutation[:eval_size]]
    
    print(f"\nEvaluating on {eval_size} samples...")
    generated_summaries = []
    
    # Generate summaries
    for conv in tqdm(conversations, desc="Generating summaries"):
        summary = summarize_PE_t5(conv, tokenizer, model)
        generated_summaries.append(summary)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_corpus_rouge(generated_summaries, ground_truth_summaries)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print some examples
    print("\nExample Summaries:")
    print("=" * 50)
    for i in range(3):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print("Original Conversation:")
        print(conversations[i][:200] + "...")  # Show first 200 chars
        print("\nGenerated Summary:")
        print(generated_summaries[i])
        print("\nGround Truth Summary:")
        print(ground_truth_summaries[i])
        print("-" * 50)
    
    return metrics


def evaluate_PE_t5():
    print("Loading T5 model and tokenizer...")
    tokenizer, model = import_baseline_t5()
    
    # Get a subset of the dataset for evaluation
    eval_size = 100 
    
    conversations = ds["train"]["conversation"][:eval_size]
    ground_truth_summaries = ds["train"]["summary"][:eval_size]

    print(f"\nEvaluating on {eval_size} samples...")
    generated_summaries = []
    
    # Generate summaries
    for conv in tqdm(conversations, desc="Generating summaries"):
        summary = summarize_PE_t5(conv, tokenizer, model)
        generated_summaries.append(summary)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_corpus_rouge(generated_summaries, ground_truth_summaries)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print some examples
    print("\nExample Summaries:")
    print("=" * 50)
    for i in range(3):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print("Original Conversation:")
        print(conversations[i][:200] + "...")  # Show first 200 chars
        print("\nGenerated Summary:")
        print(generated_summaries[i])
        print("\nGround Truth Summary:")
        print(ground_truth_summaries[i])
        print("-" * 50)
    
    return metrics
if __name__ == "__main__":
    evaluate_PE_t5()
   