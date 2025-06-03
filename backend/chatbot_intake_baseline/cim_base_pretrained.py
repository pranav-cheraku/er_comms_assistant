# backend/chatbot_intake_baseline/cim_base_pretrained.py

"""
Chatbot Intake Module Baseline Pretrained Model - T5-Base + spaCy Medical Text Summarization
Uses 80/10/10 data split, conversation->summary evaluation
Uses pretrained T5-Base model (no fine-tuning) on medical conversations to generate clinical summaries
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import spacy
import json
from tqdm import tqdm
import os
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.model_selection import train_test_split

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CIM_Baseline_Pretrained:
   """Medical Baseline: Pretrained T5-Base + spaCy with 80/10/10 data split for medical summarization"""
   
   def __init__(self):
       # Use pretrained T5-Base model for medical text summarization
       self.model_name = 't5-base'
       
       print(f"Loading pretrained T5-Base model: {self.model_name}")
       self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
       self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
       self.model.to(device)
       self.model.eval()  # Set to evaluation mode (no training)
       
       # Initialize spaCy model for entity recognition
       self.nlp = self._load_ner_model()
       
       # Results storage directory
       self.results_dir = "cim_base_pretrained_results"
       os.makedirs(self.results_dir, exist_ok=True)
       
       # Evaluation metrics storage
       self.evaluation_results = {}
       
   def _load_ner_model(self):
       """Load SciSpaCy model for named entity recognition"""
       try:
           nlp = spacy.load("en_core_sci_sm")
           print(f"Loaded SciSpaCy model: en_core_sci_sm")
           return nlp
       except (OSError, IOError):
           print("Error: SciSpaCy model not available")
           print("Install with: pip install en_core_sci_sm")
           return None
   
   def load_and_preprocess_data(self, sample_size=None):
       """Load and preprocess the medical conversation dataset from HuggingFace"""
       print("Loading medical dataset...")
       
       try:
           # Load the dataset from HuggingFace
           dataset = load_dataset("AGBonnet/augmented-clinical-notes")
           df = pd.DataFrame(dataset['train'])
           print(f"Loaded {len(df)} samples from HuggingFace dataset")
           
           # Using sample for faster evaluation if specified
           if sample_size:
               df = df.sample(n=min(sample_size, len(df)), random_state=42)
               print(f"Using sample of {len(df)} records for faster evaluation")
               
       except Exception as e:
           print(f"Error loading dataset: {e}")
           return []
       
       # Preprocess the data
       processed_data = []
       for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
           # Extract conversation and summary
           conversation = str(row.get('conversation', ''))
           summary = str(row.get('summary', ''))
           
           # Skip empty entries
           if not conversation or not summary or len(conversation.strip()) < 10:
               continue
               
           processed_data.append({
               'conversation': conversation,
               'summary': summary
           })
       
       print(f"Processed {len(processed_data)} valid samples")
       return processed_data
   
   def split_data(self, data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
       """Split data into train, dev, and test sets (80/10/10)"""
       assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
       
       # First split: separate train from (dev + test)
       train_data, temp_data = train_test_split(
           data, 
           test_size=(dev_ratio + test_ratio), 
           random_state=42
       )
       
       # Second split: separate dev from test
       dev_size = dev_ratio / (dev_ratio + test_ratio)
       dev_data, test_data = train_test_split(
           temp_data, 
           test_size=(1 - dev_size), 
           random_state=42
       )
       
       print(f"Data Split (80/10/10):")
       print(f"  Train: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
       print(f"  Dev: {len(dev_data)} samples ({len(dev_data)/len(data)*100:.1f}%)")
       print(f"  Test: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
       
       return train_data, dev_data, test_data
   
   def generate_summary(self, text, max_length=128, num_beams=4):
       """Generate medical summary using pretrained T5-Base model"""
       self.model.eval()
       input_text = f"summarize: {text}"
       
       # Tokenize input
       inputs = self.tokenizer.encode(
           input_text,
           return_tensors='pt',
           max_length=512,
           truncation=True
       ).to(device)
       
       # Generate summary using pretrained T5-Base model
       with torch.no_grad():
           summary_ids = self.model.generate(
               inputs,
               max_length=max_length,
               num_beams=num_beams,
               early_stopping=True,
               no_repeat_ngram_size=2
           )
       
       # Decode summary
       summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
       return summary
   
   def extract_medical_entities(self, text):
       """Extract entities using spaCy and structure them like the dataset JSON format"""
       if self.nlp is None:
           return {}
       
       doc = self.nlp(text)
       
       # Initialize structured format matching dataset JSON
       entities = {
           "visit motivation": "None",
           "admission": [{"reason": "None", "date": "None", "duration": "None", "care center details": "None"}],
           "patient information": {
               "age": "None", "sex": "None", "ethnicity": "None", "weight": "None", 
               "height": "None", "family medical history": "None", "recent travels": "None", 
               "socio economic context": "None", "occupation": "None"
           },
           "patient medical history": {
               "physiological context": "None", "psychological context": "None", 
               "vaccination history": "None", "allergies": "None", "exercise frequency": "None", 
               "nutrition": "None", "sexual history": "None", "alcohol consumption": "None", 
               "drug usage": "None", "smoking status": "None"
           },
           "surgeries": [{"reason": "None", "Type": "None", "time": "None", "outcome": "None", "details": "None"}],
           "symptoms": [],
           "medical examinations": [{"name": "None", "result": "None", "details": "None"}],
           "diagnosis tests": [{"test": "None", "severity": "None", "result": "None", "condition": "None", "time": "None", "details": "None"}],
           "treatments": [],
           "discharge": {"reason": "None", "referral": "None", "follow up": "None", "discharge summary": "None"}
       }
       
       # Extract entities and populate structured format
       symptoms_list = []
       treatments_list = []
       
       for ent in doc.ents:
           entity_text = ent.text.strip()
           entity_label = ent.label_
           
           # Map spaCy entities to medical structure
           if entity_label in ['CARDINAL'] and any(age_word in entity_text.lower() for age_word in ['year', 'old', 'age']):
               entities["patient information"]["age"] = entity_text
           
           elif entity_label in ['PERSON'] and any(gender in entity_text.lower() for gender in ['male', 'female', 'man', 'woman']):
               entities["patient information"]["sex"] = entity_text
           
           elif entity_label in ['QUANTITY', 'CARDINAL'] and any(unit in entity_text.lower() for unit in ['mg', 'ml', 'dose', 'tablet']):
               if not any(treatment['name'] != "None" for treatment in treatments_list):
                   treatments_list.append({
                       "name": "None",
                       "related condition": "None", 
                       "dosage": entity_text,
                       "time": "None",
                       "frequency": "None",
                       "duration": "None",
                       "reason for taking": "None",
                       "reaction to treatment": "None",
                       "details": "None"
                   })
           
           elif entity_label in ['ORG', 'PRODUCT']:
               # Potential medication
               treatments_list.append({
                   "name": entity_text,
                   "related condition": "None",
                   "dosage": "None", 
                   "time": "None",
                   "frequency": "None",
                   "duration": "None",
                   "reason for taking": "None",
                   "reaction to treatment": "None",
                   "details": "None"
               })
           
           else:
               # Default to symptoms if not categorized elsewhere
               symptoms_list.append({
                   "name of symptom": entity_text,
                   "intensity of symptom": "None",
                   "location": "None",
                   "time": "None", 
                   "temporalisation": "None",
                   "behaviours affecting the symptom": "None",
                   "details": "None"
               })
       
       # Update entities with extracted information
       if symptoms_list:
           entities["symptoms"] = symptoms_list
       else:
           entities["symptoms"] = [{"name of symptom": "None", "intensity of symptom": "None", "location": "None", "time": "None", "temporalisation": "None", "behaviours affecting the symptom": "None", "details": "None"}]
       
       if treatments_list:
           entities["treatments"] = treatments_list
       else:
           entities["treatments"] = [{"name": "None", "related condition": "None", "dosage": "None", "time": "None", "frequency": "None", "duration": "None", "reason for taking": "None", "reaction to treatment": "None", "details": "None"}]
       
       return entities
   
   def evaluate_on_test_set(self, test_data):
       """Evaluate pretrained T5-Base model on test set: conversation -> summary"""
       print("="*60)
       print("EVALUATING PRETRAINED T5-BASE MODEL")
       print("Conversation -> Summary Evaluation")
       print("="*60)
       
       # Generate predictions
       predictions = []
       references = []
       
       print(f"Generating predictions for {len(test_data)} samples...")
       
       for item in tqdm(test_data, desc="Evaluating Pretrained T5-Base"):
           # Generate summary from conversation using pretrained T5-Base
           predicted_summary = self.generate_summary(item['conversation'])
           predictions.append(predicted_summary)
           references.append(item['summary'])  # Reference summary from dataset
       
       # Calculate metrics
       results = self._calculate_metrics(predictions, references)
       results['test_samples'] = len(test_data)
       results['model_name'] = self.model_name
       results['model_type'] = 'pretrained'
       
       self.evaluation_results = results
       
       # Save results
       self._save_evaluation_results(test_data, predictions, references)
       
       return results
   
   def _calculate_metrics(self, predictions, references):
       """Calculate evaluation metrics"""
       results = {}
       
       # ROUGE Scores
       print("Calculating ROUGE scores...")
       scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
       rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
       
       for pred, ref in zip(predictions, references):
           scores = scorer.score(ref, pred)
           for metric in rouge_scores:
               rouge_scores[metric].append(scores[metric].fmeasure)
       
       for metric in rouge_scores:
           results[f'{metric}_mean'] = np.mean(rouge_scores[metric])
           results[f'{metric}_std'] = np.std(rouge_scores[metric])
       
       # BERTScore
       print("Calculating BERTScore...")
       try:
           P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
           results['bertscore_precision'] = P.mean().item()
           results['bertscore_recall'] = R.mean().item()
           results['bertscore_f1'] = F1.mean().item()
       except Exception as e:
           print(f"BERTScore calculation failed: {e}")
       
       # Summary statistics
       pred_lengths = [len(p.split()) for p in predictions]
       ref_lengths = [len(r.split()) for r in references]
       
       results['avg_pred_length'] = np.mean(pred_lengths)
       results['avg_ref_length'] = np.mean(ref_lengths)
       results['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths)
       
       return results
   
   def _save_evaluation_results(self, test_data, predictions, references):
       """Save evaluation results"""
       # Detailed results
       detailed_results = []
       for i, item in enumerate(test_data):
           detailed_results.append({
               'conversation': item['conversation'],
               'reference_summary': references[i],
               'predicted_summary': predictions[i]
           })
       
       df_results = pd.DataFrame(detailed_results)
       results_file = os.path.join(self.results_dir, 'detailed_results.csv')
       df_results.to_csv(results_file, index=False)
       
       # Summary metrics
       metrics_file = os.path.join(self.results_dir, 'evaluation_metrics.json')
       
       with open(metrics_file, 'w') as f:
           json.dump(self.evaluation_results, f, indent=2)
       
       print(f"Detailed results saved to {results_file}")
       print(f"Metrics saved to {metrics_file}")
   
   def print_results(self):
       """Print comprehensive results"""
       print("\n" + "="*60)
       print("PRETRAINED MEDICAL BASELINE EVALUATION RESULTS")
       print("="*60)
       
       if self.evaluation_results:
           print("\nMODEL INFORMATION:")
           print(f"  Model: {self.evaluation_results['model_name']}")
           print(f"  Type: {self.evaluation_results['model_type']} (no fine-tuning)")
           print(f"  Test Samples: {self.evaluation_results['test_samples']}")
           
           print(f"\n  ROUGE SCORES:")
           print(f"    ROUGE-1: {self.evaluation_results.get('rouge1_mean', 0):.4f} (±{self.evaluation_results.get('rouge1_std', 0):.4f})")
           print(f"    ROUGE-2: {self.evaluation_results.get('rouge2_mean', 0):.4f} (±{self.evaluation_results.get('rouge2_std', 0):.4f})")
           print(f"    ROUGE-L: {self.evaluation_results.get('rougeL_mean', 0):.4f} (±{self.evaluation_results.get('rougeL_std', 0):.4f})")
           
           print(f"\n  BERTSCORE:")
           print(f"    Precision: {self.evaluation_results.get('bertscore_precision', 0):.4f}")
           print(f"    Recall: {self.evaluation_results.get('bertscore_recall', 0):.4f}")
           print(f"    F1: {self.evaluation_results.get('bertscore_f1', 0):.4f}")
           
           print(f"\n  SUMMARY STATISTICS:")
           print(f"    Avg Predicted Length: {self.evaluation_results['avg_pred_length']:.1f} words")
           print(f"    Avg Reference Length: {self.evaluation_results['avg_ref_length']:.1f} words")
           print(f"    Length Ratio: {self.evaluation_results['length_ratio']:.3f}")
       
       print(f"\nResults saved to: {self.results_dir}")

def main():
   """Run Medical Baseline evaluation with pretrained model"""
   print("="*60)
   print("MEDICAL BASELINE")
   print("Pretrained T5-Base + spaCy (80/10/10 Split)")
   print("Conversation -> Summary Evaluation")
   print("No Fine-tuning - Using Pretrained Model Only")
   print("="*60)
   
   # Initialize baseline
   baseline = CIM_Baseline_Pretrained()
   
   # Load and split data
   data = baseline.load_and_preprocess_data(sample_size=1000)  # Use only 1000 samples for speed
   if not data:
       print("Error: No data loaded. Exiting.")
       return None
   
   train_data, dev_data, test_data = baseline.split_data(data)
   
   # No training - directly evaluate on test set using pretrained model
   print("\nSkipping training - using pretrained T5-Base model")
   
   # Evaluate on test set
   baseline.evaluate_on_test_set(test_data)
   
   # Print comprehensive results
   baseline.print_results()
   
   return baseline

if __name__ == "__main__":
   baseline = main()