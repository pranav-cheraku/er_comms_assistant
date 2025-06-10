# backend/chatbot_intake_baseline/cim_base_tuned.py

"""
Chatbot Intake Module Basline Tuned Model - T5-Base + spaCy Medical Text Summarization
Uses 80/10/10 data split, conversation->summary evaluation
Fine-tuned T5-Base model on medical conversations to generate clinical summaries

This module implements a medical text summarization system that:
1. Fine-tunes a T5-Base transformer model on medical conversation data
2. Uses spaCy for named entity recognition to extract medical entities
3. Evaluates performance using ROUGE and BERTScore metrics
4. Follows a standard 80/10/10 train/dev/test split
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

# GPU/CPU device detection, prioritizes CUDA for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MedicalDataset(Dataset):
   """
   Custom PyTorch Dataset class for medical conversations and summaries
   
   This class handles the tokenization and formatting of medical data for T5 model training.
   It prepares conversation-summary pairs for the fine-tuning process.
   """
   
   def __init__(self, tokenizer, data, max_source_length=512, max_target_length=128):
       """
       Initialize the dataset with tokenizer and length constraints
       
       Args:
           tokenizer: T5Tokenizer for encoding text
           data: List of conversation-summary pairs
           max_source_length: Maximum tokens for input conversations
           max_target_length: Maximum tokens for target summaries
       """
       self.tokenizer = tokenizer
       self.data = data
       self.max_source_length = max_source_length
       self.max_target_length = max_target_length
   
   def __len__(self):
       """Return the total number of samples in the dataset"""
       return len(self.data)
   
   def __getitem__(self, idx):
       """
       Get a single training sample with proper T5 formatting
       
       Returns tokenized inputs and labels for T5 training:
       - Source: "summarize: {conversation}" (T5 task prefix)
       - Target: clinical summary
       """
       item = self.data[idx]
       
       # Input: conversation, Target: summary
       source_text = f"summarize: {item['conversation']}" 
       target_text = item['summary']
       
       # Tokenize inputs with padding and truncation for consistent batch sizes
       source_encoding = self.tokenizer(
           source_text,
           max_length=self.max_source_length,
           padding='max_length',  # Pad to max_length for batch processing
           truncation=True,       # Truncate long conversations
           return_tensors='pt'
       )
       
       # Tokenize target summaries
       target_encoding = self.tokenizer(
           target_text,
           max_length=self.max_target_length,
           padding='max_length',
           truncation=True,
           return_tensors='pt'
       )
       
       # Return formatted tensors for PyTorch DataLoader
       return {
           'input_ids': source_encoding['input_ids'].flatten(),
           'attention_mask': source_encoding['attention_mask'].flatten(),
           'labels': target_encoding['input_ids'].flatten()  # T5 uses labels for target sequences
       }

class CIM_Baseline_Tuned:
   """
   Medical Baseline: T5-Base + spaCy with 80/10/10 data split for medical summarization
   
   This class implements the complete pipeline for medical text summarization:
   1. Data loading and preprocessing from HuggingFace dataset
   2. T5-Base model fine-tuning on medical conversations
   3. spaCy-based named entity recognition for medical concepts
   4. Comprehensive evaluation using multiple metrics
   """
   
   def __init__(self):
       """
       Initialize the medical summarization system with T5-Base and spaCy models
       """
       # Use T5-Base model for medical text summarization
       self.model_name = 't5-base'  # Pre-trained T5-Base from HuggingFace
       
       print(f"Loading T5-Base model: {self.model_name}")
       # Load tokenizer and model from HuggingFace transformers
       self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
       self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
       self.model.to(device)  # Move model to GPU if available
       
       # Initialize spaCy model for entity recognition
       self.nlp = self._load_ner_model()
       
       # Results storage directory for saving model outputs and metrics
       self.results_dir = "cim_base_tuned_results"
       os.makedirs(self.results_dir, exist_ok=True)
       
       # Training and evaluation metrics storage for analysis
       self.training_results = {}
       self.evaluation_results = {}
       
   def _load_ner_model(self):
       """
       Load SciSpaCy model for named entity recognition in medical text
       
       SciSpaCy is specifically designed for biomedical/clinical text processing
       and can identify medical entities like diseases, treatments, etc.
       """
       try:
           nlp = spacy.load("en_core_sci_sm")  # Scientific/medical spaCy model
           print(f"Loaded SciSpaCy model: en_core_sci_sm")
           return nlp
       except (OSError, IOError):
           print("Error: SciSpaCy model not available")
           print("Install with: pip install en_core_sci_sm")
           return None
   
   def load_and_preprocess_data(self, sample_size=None):
       """
       Load and preprocess the medical conversation dataset from HuggingFace
       
       This method:
       1. Downloads the AGBonnet/augmented-clinical-notes dataset
       2. Filters out invalid/empty entries
       3. Optionally samples data for faster training during development
       
       Args:
           sample_size: Optional limit on number of samples for faster experimentation
       """
       print("Loading medical dataset...")
       
       try:
           # Load the dataset from HuggingFace hub
           dataset = load_dataset("AGBonnet/augmented-clinical-notes")
           df = pd.DataFrame(dataset['train'])  # Convert to pandas for easier manipulation
           print(f"Loaded {len(df)} samples from HuggingFace dataset")
           
           # Using sample for faster T5-Base training if specified
           if sample_size:
               df = df.sample(n=min(sample_size, len(df)), random_state=42)
               print(f"Using sample of {len(df)} records for faster T5-Base training")
               
       except Exception as e:
           print(f"Error loading dataset: {e}")
           return []
       
       # Preprocess the data to extract conversation-summary pairs
       processed_data = []
       for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
           # Extract conversation and summary fields from dataset
           conversation = str(row.get('conversation', ''))
           summary = str(row.get('summary', ''))
           
           # Skip empty entries to ensure data quality
           if not conversation or not summary or len(conversation.strip()) < 10:
               continue
               
           processed_data.append({
               'conversation': conversation,
               'summary': summary
           })
       
       print(f"Processed {len(processed_data)} valid samples")
       return processed_data
   
   def split_data(self, data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
       """
       Split data into train, dev, and test sets using standard 80/10/10 methodology
       
       This follows machine learning best practices:
       - Train (80%): For model parameter learning
       - Dev (10%): For hyperparameter tuning and validation during training
       - Test (10%): For final unbiased evaluation
       
       Args:
           data: Preprocessed conversation-summary pairs
           train_ratio: Proportion for training (0.8 = 80%)
           dev_ratio: Proportion for development/validation (0.1 = 10%)
           test_ratio: Proportion for testing (0.1 = 10%)
       """
       assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
       
       # First split: separate train from (dev + test)
       train_data, temp_data = train_test_split(
           data, 
           test_size=(dev_ratio + test_ratio),  # Combined dev+test size
           random_state=42  # Fixed seed for reproducibility
       )
       
       # Second split: separate dev from test
       dev_size = dev_ratio / (dev_ratio + test_ratio)  # Relative proportion within temp_data
       dev_data, test_data = train_test_split(
           temp_data, 
           test_size=(1 - dev_size), 
           random_state=42
       )
       
       # Print split statistics for verification
       print(f"Data Split (80/10/10):")
       print(f"  Train: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
       print(f"  Dev: {len(dev_data)} samples ({len(dev_data)/len(data)*100:.1f}%)")
       print(f"  Test: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
       
       return train_data, dev_data, test_data
   
   def create_datasets(self, train_data, dev_data, test_data):
       """
       Create PyTorch datasets for T5-Base training from split data
       
       Converts raw text data into tokenized PyTorch Dataset objects
       that can be used with DataLoader for batch processing
       """
       train_dataset = MedicalDataset(self.tokenizer, train_data)
       dev_dataset = MedicalDataset(self.tokenizer, dev_data)
       test_dataset = MedicalDataset(self.tokenizer, test_data)
       
       return train_dataset, dev_dataset, test_dataset
   
   def train_model(self, train_dataset, dev_dataset, num_epochs=3, batch_size=4, learning_rate=5e-5):
       """
       Train the Clinical T5 model using fine-tuning approach
       
       This method implements the complete training loop:
       1. Sets up data loaders for batch processing
       2. Configures AdamW optimizer (recommended for transformers)
       3. Runs training epochs with loss calculation and backpropagation
       4. Validates on development set after each epoch
       5. Saves the fine-tuned model
       
       Args:
           train_dataset: Tokenized training data
           dev_dataset: Tokenized validation data  
           num_epochs: Number of training epochs (3 is typical for fine-tuning)
           batch_size: Samples per batch (4 works well for T5-base on typical GPUs)
           learning_rate: Learning rate for AdamW (5e-5 is standard for T5 fine-tuning)
       """
       print("="*60)
       print("STARTING T5-BASE FINE-TUNING")
       print("="*60)
       
       # Create data loaders for T5-Base training with batch processing
       train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
       
       # Setup AdamW optimizer for T5-Base parameters (recommended for transformers)
       optimizer = AdamW(self.model.parameters(), lr=learning_rate)
       
       # Set model to training mode (enables dropout, batch norm updates)
       self.model.train()
       
       total_train_loss = 0
       
       # Print training configuration for reproducibility
       print(f"Fine-tuning T5-Base on {len(train_dataset)} samples...")
       print(f"Validating on {len(dev_dataset)} samples...")
       print(f"Training for {num_epochs} epochs...")
       print(f"Model: {self.model_name}")
       print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
       
       # Main training loop
       for epoch in range(num_epochs):
           print(f"\nEpoch {epoch + 1}/{num_epochs}")
           
           # Training phase for current epoch
           epoch_loss = 0
           train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
           
           for batch_idx, batch in enumerate(train_pbar):
               # Move batch tensors to GPU/CPU device
               input_ids = batch['input_ids'].to(device)
               attention_mask = batch['attention_mask'].to(device)
               labels = batch['labels'].to(device)
               
               # Forward pass through T5 model
               outputs = self.model(
                   input_ids=input_ids,
                   attention_mask=attention_mask,
                   labels=labels  # T5 computes loss automatically when labels provided
               )
               
               loss = outputs.loss  # Cross-entropy loss for sequence generation
               epoch_loss += loss.item()
               
               # Backward pass and parameter updates
               optimizer.zero_grad()  # Clear gradients from previous step
               loss.backward()        # Compute gradients via backpropagation
               optimizer.step()       # Update model parameters
               
               # Update progress bar with current loss
               train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
               
               # Log every 100 batches for monitoring training progress
               if batch_idx > 0 and batch_idx % 100 == 0:
                   avg_loss = epoch_loss / (batch_idx + 1)
                   print(f"  Batch {batch_idx}, Avg Loss: {avg_loss:.4f}")
           
           # Calculate and display epoch statistics
           avg_epoch_loss = epoch_loss / len(train_loader)
           total_train_loss += avg_epoch_loss
           print(f"  Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")
           
           # Validation phase to monitor overfitting
           if len(dev_loader) > 0:
               val_loss = self._validate(dev_loader)
               print(f"  Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")
       
       # Calculate final training statistics
       final_loss = total_train_loss / num_epochs
       
       # Save training results for analysis
       self.training_results = {
           'final_train_loss': final_loss,
           'train_samples': len(train_dataset),
           'dev_samples': len(dev_dataset),
           'epochs': num_epochs,
           'model_name': self.model_name
       }
       
       # Save the fine-tuned T5-Base model to disk
       model_dir = os.path.join(self.results_dir, 'model')
       os.makedirs(model_dir, exist_ok=True)
       self.model.save_pretrained(model_dir)      # Save model weights and config
       self.tokenizer.save_pretrained(model_dir)  # Save tokenizer for inference
       
       print(f"T5-Base fine-tuning completed")
       print(f"Final average training loss: {final_loss:.4f}")
       print(f"Fine-tuned T5-Base model saved to {model_dir}")
   
   def _validate(self, dev_loader):
       """
       Run validation on development set during T5-Base training
       
       This method evaluates the model on the development set without updating parameters.
       Used to monitor overfitting and select the best model checkpoint.
       """
       self.model.eval()  # Set to evaluation mode (disables dropout)
       total_val_loss = 0
       
       # Disable gradient computation for faster validation
       with torch.no_grad():
           for batch in dev_loader:
               # Move validation batch to device
               input_ids = batch['input_ids'].to(device)
               attention_mask = batch['attention_mask'].to(device)
               labels = batch['labels'].to(device)
               
               # Forward pass only (no backpropagation)
               outputs = self.model(
                   input_ids=input_ids,
                   attention_mask=attention_mask,
                   labels=labels
               )
               
               total_val_loss += outputs.loss.item()
       
       # Return to training mode for next epoch
       self.model.train()
       return total_val_loss / len(dev_loader)
   
   def generate_summary(self, text, max_length=128, num_beams=4):
       """
       Generate medical summary using fine-tuned T5-Base model
       
       This method performs inference with the trained model to generate
       clinical summaries from patient conversations.
       
       Args:
           text: Input conversation text
           max_length: Maximum length of generated summary (128 tokens)
           num_beams: Beam search width for better generation quality
       """
       self.model.eval()  # Set to evaluation mode
       input_text = f"summarize: {text}"  # Add T5 task prefix
       
       # Tokenize input conversation
       inputs = self.tokenizer.encode(
           input_text,
           return_tensors='pt',
           max_length=512,        # Match training max_source_length
           truncation=True
       ).to(device)
       
       # Generate summary using fine-tuned T5-Base model
       with torch.no_grad():
           summary_ids = self.model.generate(
               inputs,
               max_length=max_length,
               num_beams=num_beams,       # Beam search for better quality
               early_stopping=True,      # Stop when EOS token generated
               no_repeat_ngram_size=2    # Prevent repetitive phrases
           )
       
       # Decode generated tokens back to text
       summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
       return summary
   
   def extract_medical_entities(self, text):
       """
       Extract entities using spaCy and structure them like the dataset JSON format
       
       This method uses the SciSpaCy model to identify medical entities in text
       and organizes them into a structured format matching the clinical dataset schema.
       This enables extraction of:
       - Patient information (age, sex, etc.)
       - Medical history
       - Symptoms and treatments
       - Diagnostic tests and results
       """
       if self.nlp is None:
           return {}
       
       # Process text with spaCy NLP pipeline
       doc = self.nlp(text)
       
       # Initialize structured format matching dataset JSON schema
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
       
       # Process each named entity identified by spaCy
       for ent in doc.ents:
           entity_text = ent.text.strip()
           entity_label = ent.label_
           
           # Map spaCy entities to medical structure based on context and label
           if entity_label in ['CARDINAL'] and any(age_word in entity_text.lower() for age_word in ['year', 'old', 'age']):
               entities["patient information"]["age"] = entity_text
           
           elif entity_label in ['PERSON'] and any(gender in entity_text.lower() for gender in ['male', 'female', 'man', 'woman']):
               entities["patient information"]["sex"] = entity_text
           
           elif entity_label in ['QUANTITY', 'CARDINAL'] and any(unit in entity_text.lower() for unit in ['mg', 'ml', 'dose', 'tablet']):
               # Dosage information for treatments
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
               # Potential medication names
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
       
       # Update entities with extracted information or default empty structures
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
       """
       Evaluate fine-tuned T5-Base model on test set: conversation -> summary
       
       This method performs comprehensive evaluation of the trained model:
       1. Generates predictions for all test samples
       2. Calculates multiple evaluation metrics (ROUGE, BERTScore)
       3. Analyzes prediction quality and characteristics
       4. Saves detailed results for further analysis
       """
       print("="*60)
       print("EVALUATING FINE-TUNED T5-BASE MODEL")
       print("Conversation -> Summary Evaluation")
       print("="*60)
       
       # Generate predictions for all test samples
       predictions = []
       references = []
       
       print(f"Generating predictions for {len(test_data)} samples...")
       
       for item in tqdm(test_data, desc="Evaluating T5-Base"):
           # Generate summary from conversation using fine-tuned T5-Base
           predicted_summary = self.generate_summary(item['conversation'])
           predictions.append(predicted_summary)
           references.append(item['summary'])  # Reference summary from dataset
       
       # Calculate comprehensive evaluation metrics
       results = self._calculate_metrics(predictions, references)
       results['test_samples'] = len(test_data)
       
       self.evaluation_results = results
       
       # Save detailed results and metrics to files
       self._save_evaluation_results(test_data, predictions, references)
       
       return results
   
   def _calculate_metrics(self, predictions, references):
       """
       Calculate evaluation metrics for summarization quality
       
       Computes multiple metrics to assess different aspects of summary quality:
       - ROUGE: Lexical overlap between predicted and reference summaries
       - BERTScore: Semantic similarity using contextual embeddings
       - Length statistics: Analysis of summary length characteristics
       """
       results = {}
       
       # ROUGE Scores - measure n-gram overlap between prediction and reference
       print("Calculating ROUGE scores...")
       scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
       rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
       
       for pred, ref in zip(predictions, references):
           scores = scorer.score(ref, pred)
           for metric in rouge_scores:
               rouge_scores[metric].append(scores[metric].fmeasure)  # F1 score
       
       # Calculate mean and standard deviation for each ROUGE metric
       for metric in rouge_scores:
           results[f'{metric}_mean'] = np.mean(rouge_scores[metric])
           results[f'{metric}_std'] = np.std(rouge_scores[metric])
       
       # BERTScore - semantic similarity using contextual embeddings
       print("Calculating BERTScore...")
       try:
           P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
           results['bertscore_precision'] = P.mean().item()
           results['bertscore_recall'] = R.mean().item()
           results['bertscore_f1'] = F1.mean().item()
       except Exception as e:
           print(f"BERTScore calculation failed: {e}")
       
       # Summary statistics - analyze length characteristics
       pred_lengths = [len(p.split()) for p in predictions]  # Word counts
       ref_lengths = [len(r.split()) for r in references]
       
       results['avg_pred_length'] = np.mean(pred_lengths)
       results['avg_ref_length'] = np.mean(ref_lengths)
       results['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths)
       
       return results
   
   def _save_evaluation_results(self, test_data, predictions, references):
       """
       Save evaluation results to files for detailed analysis
       
       Creates two output files:
       1. Detailed results CSV with all predictions and references
       2. Summary metrics JSON with training and evaluation statistics
       """
       # Detailed results with individual predictions
       detailed_results = []
       for i, item in enumerate(test_data):
           detailed_results.append({
               'conversation': item['conversation'],
               'reference_summary': references[i],
               'predicted_summary': predictions[i]
           })
       
       # Save detailed results as CSV for manual inspection
       df_results = pd.DataFrame(detailed_results)
       results_file = os.path.join(self.results_dir, 'detailed_results.csv')
       df_results.to_csv(results_file, index=False)
       
       # Summary metrics combining training and evaluation results
       metrics_file = os.path.join(self.results_dir, 'evaluation_metrics.json')
       combined_results = {
           'training_results': self.training_results,
           'evaluation_results': self.evaluation_results
       }
       
       # Save metrics as JSON for programmatic access
       with open(metrics_file, 'w') as f:
           json.dump(combined_results, f, indent=2)
       
       print(f"Detailed results saved to {results_file}")
       print(f"Metrics saved to {metrics_file}")
   
   def print_results(self):
       """
       Print comprehensive results in a formatted, readable manner
       
       Displays all training and evaluation metrics in an organized format
       for easy interpretation of model performance.
       """
       print("\n" + "="*60)
       print("MEDICAL BASELINE EVALUATION RESULTS")
       print("="*60)
       
       # Training results section
       if self.training_results:
           print("\nTRAINING RESULTS:")
           print(f"  Model: {self.training_results['model_name']}")
           print(f"  Final Training Loss: {self.training_results['final_train_loss']:.4f}")
           print(f"  Training Samples: {self.training_results['train_samples']}")
           print(f"  Dev Samples: {self.training_results['dev_samples']}")
           print(f"  Epochs: {self.training_results['epochs']}")
       
       # Evaluation results section
       if self.evaluation_results:
           print("\nEVALUATION RESULTS:")
           print(f"  Test Samples: {self.evaluation_results['test_samples']}")
           
           # ROUGE scores with confidence intervals
           print(f"\n  ROUGE SCORES:")
           print(f"    ROUGE-1: {self.evaluation_results.get('rouge1_mean', 0):.4f} (±{self.evaluation_results.get('rouge1_std', 0):.4f})")
           print(f"    ROUGE-2: {self.evaluation_results.get('rouge2_mean', 0):.4f} (±{self.evaluation_results.get('rouge2_std', 0):.4f})")
           print(f"    ROUGE-L: {self.evaluation_results.get('rougeL_mean', 0):.4f} (±{self.evaluation_results.get('rougeL_std', 0):.4f})")
           
           # BERTScore semantic similarity metrics
           print(f"\n  BERTSCORE:")
           print(f"    Precision: {self.evaluation_results.get('bertscore_precision', 0):.4f}")
           print(f"    Recall: {self.evaluation_results.get('bertscore_recall', 0):.4f}")
           print(f"    F1: {self.evaluation_results.get('bertscore_f1', 0):.4f}")
           
           # Summary length analysis
           print(f"\n  SUMMARY STATISTICS:")
           print(f"    Avg Predicted Length: {self.evaluation_results['avg_pred_length']:.1f} words")
           print(f"    Avg Reference Length: {self.evaluation_results['avg_ref_length']:.1f} words")
           print(f"    Length Ratio: {self.evaluation_results['length_ratio']:.3f}")
       
       print(f"\nResults saved to: {self.results_dir}")

def main():
   """
   Run Medical Baseline evaluation - main execution function
   
   This function orchestrates the complete pipeline:
   1. Initialize the baseline model and components
   2. Load and preprocess medical conversation data
   3. Split data using standard methodology (80/10/10)
   4. Fine-tune T5-Base model on medical conversations
   5. Evaluate performance on test set
   6. Display and save comprehensive results
   """
   print("="*60)
   print("MEDICAL BASELINE")
   print("T5-Base + spaCy (80/10/10 Split)")
   print("Conversation -> Summary Evaluation")
   print("="*60)
   
   # Initialize baseline model with T5-Base and spaCy
   baseline = CIM_Baseline_Tuned()
   
   # Load and split data -> Fast version for development/testing
   data = baseline.load_and_preprocess_data(sample_size=1000)  # Use only 1000 samples for speed
   if not data:
       print("Error: No data loaded. Exiting.")
       return None
   
   # Split data using standard 80/10/10 methodology
   train_data, dev_data, test_data = baseline.split_data(data)
   
   # Create PyTorch datasets for model training
   train_dataset, dev_dataset, test_dataset = baseline.create_datasets(
       train_data, dev_data, test_data
   )
   
   # Train the model -> Fast settings for development
   baseline.train_model(train_dataset, dev_dataset, num_epochs=3, batch_size=8)
   
   # Evaluate on test set using multiple metrics
   baseline.evaluate_on_test_set(test_data)
   
   # Print comprehensive results summary
   baseline.print_results()
   
   return baseline

# Entry point for script execution
if __name__ == "__main__":
   baseline = main()