# backend/chatbot_intake_baseline/cim_base_tuned.py

"""
Chatbot Intake Module Basline Tuned Model - T5-Base + spaCy Medical Text Summarization
Uses 80/10/10 data split, conversation->summary evaluation
Fine-tuned T5-Base model on medical conversations to generate clinical summaries
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import spacy
import json
from tqdm import tqdm
import os
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MedicalDataset(Dataset):
   """Custom dataset class for medical conversations and summaries"""
   
   def __init__(self, tokenizer, data, max_source_length=512, max_target_length=128):
       self.tokenizer = tokenizer
       self.data = data
       self.max_source_length = max_source_length
       self.max_target_length = max_target_length
   
   def __len__(self):
       return len(self.data)
   
   def __getitem__(self, idx):
       item = self.data[idx]
       
       # Input: conversation, Target: summary
       source_text = f"summarize: {item['conversation']}"
       target_text = item['summary']
       
       # Tokenize inputs
       source_encoding = self.tokenizer(
           source_text,
           max_length=self.max_source_length,
           padding='max_length',
           truncation=True,
           return_tensors='pt'
       )
       
       target_encoding = self.tokenizer(
           target_text,
           max_length=self.max_target_length,
           padding='max_length',
           truncation=True,
           return_tensors='pt'
       )
       
       return {
           'input_ids': source_encoding['input_ids'].flatten(),
           'attention_mask': source_encoding['attention_mask'].flatten(),
           'labels': target_encoding['input_ids'].flatten()
       }

class CIM_Baseline_Tuned:
   """Medical Baseline: T5-Base + spaCy with 80/10/10 data split for medical summarization"""
   
   def __init__(self):
       # Use T5-Base model for medical text summarization
       self.model_name = 't5-base'
       
       print(f"Loading T5-Base model: {self.model_name}")
       self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
       self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
       self.model.to(device)
       
       # Initialize spaCy model for entity recognition
       self.nlp = self._load_ner_model()
       
       # Results storage directory
       self.results_dir = "cim_base_tuned_results"
       os.makedirs(self.results_dir, exist_ok=True)
       
       # Training and evaluation metrics storage
       self.training_results = {}
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
           
           # Using sample for faster T5-Base training if specified
           if sample_size:
               df = df.sample(n=min(sample_size, len(df)), random_state=42)
               print(f"Using sample of {len(df)} records for faster T5-Base training")
               
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
   
   def create_datasets(self, train_data, dev_data, test_data):
       """Create PyTorch datasets for T5-Base training"""
       train_dataset = MedicalDataset(self.tokenizer, train_data)
       dev_dataset = MedicalDataset(self.tokenizer, dev_data)
       test_dataset = MedicalDataset(self.tokenizer, test_data)
       
       return train_dataset, dev_dataset, test_dataset
   
   def train_model(self, train_dataset, dev_dataset, num_epochs=3, batch_size=4, learning_rate=5e-5):
       """Train the Clinical T5 model"""
       print("="*60)
       print("STARTING T5-BASE FINE-TUNING")
       print("="*60)
       
       # Create data loaders for T5-Base training
       train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
       
       # Setup AdamW optimizer for T5-Base parameters
       optimizer = AdamW(self.model.parameters(), lr=learning_rate)
       
       self.model.train()
       
       total_train_loss = 0
       
       print(f"Fine-tuning T5-Base on {len(train_dataset)} samples...")
       print(f"Validating on {len(dev_dataset)} samples...")
       print(f"Training for {num_epochs} epochs...")
       print(f"Model: {self.model_name}")
       print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
       
       for epoch in range(num_epochs):
           print(f"\nEpoch {epoch + 1}/{num_epochs}")
           
           # Training phase
           epoch_loss = 0
           train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
           
           for batch_idx, batch in enumerate(train_pbar):
               # Move batch to device
               input_ids = batch['input_ids'].to(device)
               attention_mask = batch['attention_mask'].to(device)
               labels = batch['labels'].to(device)
               
               # Forward pass
               outputs = self.model(
                   input_ids=input_ids,
                   attention_mask=attention_mask,
                   labels=labels
               )
               
               loss = outputs.loss
               epoch_loss += loss.item()
               
               # Backward pass
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               
               # Update progress bar
               train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
               
               # Log every 100 batches
               if batch_idx > 0 and batch_idx % 100 == 0:
                   avg_loss = epoch_loss / (batch_idx + 1)
                   print(f"  Batch {batch_idx}, Avg Loss: {avg_loss:.4f}")
           
           avg_epoch_loss = epoch_loss / len(train_loader)
           total_train_loss += avg_epoch_loss
           print(f"  Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")
           
           # Validation phase
           if len(dev_loader) > 0:
               val_loss = self._validate(dev_loader)
               print(f"  Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")
       
       final_loss = total_train_loss / num_epochs
       
       # Save training results
       self.training_results = {
           'final_train_loss': final_loss,
           'train_samples': len(train_dataset),
           'dev_samples': len(dev_dataset),
           'epochs': num_epochs,
           'model_name': self.model_name
       }
       
       # Save the fine-tuned T5-Base model
       model_dir = os.path.join(self.results_dir, 'model')
       os.makedirs(model_dir, exist_ok=True)
       self.model.save_pretrained(model_dir)
       self.tokenizer.save_pretrained(model_dir)
       
       print(f"T5-Base fine-tuning completed")
       print(f"Final average training loss: {final_loss:.4f}")
       print(f"Fine-tuned T5-Base model saved to {model_dir}")
   
   def _validate(self, dev_loader):
       """Run validation on development set during T5-Base training"""
       self.model.eval()
       total_val_loss = 0
       
       with torch.no_grad():
           for batch in dev_loader:
               input_ids = batch['input_ids'].to(device)
               attention_mask = batch['attention_mask'].to(device)
               labels = batch['labels'].to(device)
               
               outputs = self.model(
                   input_ids=input_ids,
                   attention_mask=attention_mask,
                   labels=labels
               )
               
               total_val_loss += outputs.loss.item()
       
       self.model.train()
       return total_val_loss / len(dev_loader)
   
   def generate_summary(self, text, max_length=128, num_beams=4):
       """Generate medical summary using fine-tuned T5-Base model"""
       self.model.eval()
       input_text = f"summarize: {text}"
       
       # Tokenize input
       inputs = self.tokenizer.encode(
           input_text,
           return_tensors='pt',
           max_length=512,
           truncation=True
       ).to(device)
       
       # Generate summary using fine-tuned T5-Base model
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
       """Evaluate fine-tuned T5-Base model on test set: conversation -> summary"""
       print("="*60)
       print("EVALUATING FINE-TUNED T5-BASE MODEL")
       print("Conversation -> Summary Evaluation")
       print("="*60)
       
       # Generate predictions
       predictions = []
       references = []
       
       print(f"Generating predictions for {len(test_data)} samples...")
       
       for item in tqdm(test_data, desc="Evaluating T5-Base"):
           # Generate summary from conversation using fine-tuned T5-Base
           predicted_summary = self.generate_summary(item['conversation'])
           predictions.append(predicted_summary)
           references.append(item['summary'])  # Reference summary from dataset
       
       # Calculate metrics
       results = self._calculate_metrics(predictions, references)
       results['test_samples'] = len(test_data)
       
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
       combined_results = {
           'training_results': self.training_results,
           'evaluation_results': self.evaluation_results
       }
       
       with open(metrics_file, 'w') as f:
           json.dump(combined_results, f, indent=2)
       
       print(f"Detailed results saved to {results_file}")
       print(f"Metrics saved to {metrics_file}")
   
   def print_results(self):
       """Print comprehensive results"""
       print("\n" + "="*60)
       print("MEDICAL BASELINE EVALUATION RESULTS")
       print("="*60)
       
       if self.training_results:
           print("\nTRAINING RESULTS:")
           print(f"  Model: {self.training_results['model_name']}")
           print(f"  Final Training Loss: {self.training_results['final_train_loss']:.4f}")
           print(f"  Training Samples: {self.training_results['train_samples']}")
           print(f"  Dev Samples: {self.training_results['dev_samples']}")
           print(f"  Epochs: {self.training_results['epochs']}")
       
       if self.evaluation_results:
           print("\nEVALUATION RESULTS:")
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
   """Run Medical Baseline evaluation"""
   print("="*60)
   print("MEDICAL BASELINE")
   print("T5-Base + spaCy (80/10/10 Split)")
   print("Conversation -> Summary Evaluation")
   print("="*60)
   
   # Initialize baseline
   baseline = CIM_Baseline_Tuned()
   
   # Load and split data -> Fast version
   data = baseline.load_and_preprocess_data(sample_size=1000)  # Use only 1000 samples for speed
   if not data:
       print("Error: No data loaded. Exiting.")
       return None
   
   train_data, dev_data, test_data = baseline.split_data(data)
   
   # Create datasets
   train_dataset, dev_dataset, test_dataset = baseline.create_datasets(
       train_data, dev_data, test_data
   )
   
   # Train the model -> Fast settings
   baseline.train_model(train_dataset, dev_dataset, num_epochs=3, batch_size=8)
   
   # Evaluate on test set
   baseline.evaluate_on_test_set(test_data)
   
   # Print comprehensive results
   baseline.print_results()
   
   return baseline

if __name__ == "__main__":
   baseline = main()