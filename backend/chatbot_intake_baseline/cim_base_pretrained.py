# backend/chatbot_intake_baseline/cim_base_pretrained.py

"""
Chatbot Intake Module Baseline Pretrained Model - T5-Base + spaCy Medical Text Summarization
Uses 80/10/10 data split, conversation->summary evaluation
Uses pretrained T5-Base model (no fine-tuning) on medical conversations to generate clinical summaries

This baseline model serves as a benchmark for evaluating the performance of pretrained transformer models
on medical text summarization tasks without any domain-specific fine-tuning.
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

# Check if CUDA is available for GPU acceleration
# This is important for transformer models which are computationally intensive
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CIM_Baseline_Pretrained:
   """
   Medical Baseline: Pretrained T5-Base + spaCy with 80/10/10 data split for medical summarization
   
   This class implements a baseline approach for medical text summarization using:
   1. Pretrained T5-Base model for text-to-text generation (summarization)
   2. SciSpaCy for Named Entity Recognition in medical texts
   3. Standard 80/10/10 train/dev/test split for evaluation
   
   The model is NOT fine-tuned on medical data, serving as a baseline to compare against
   domain-specific models.
   """
   
   def __init__(self):
       """
       Initialize the baseline model with pretrained components
       
       Components initialized:
       - T5-Base tokenizer and model for text summarization
       - SciSpaCy model for medical named entity recognition
       - Results directory for saving evaluation outputs
       """
       # Use pretrained T5-Base model for medical text summarization
       # T5 is a text-to-text transformer that can handle summarization by prefixing input with "summarize:"
       self.model_name = 't5-base'
       
       print(f"Loading pretrained T5-Base model: {self.model_name}")
       # Load tokenizer to convert text to tokens that the model can process
       self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
       # Load the actual T5 model for conditional generation (summarization)
       self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
       # Move model to GPU if available for faster inference
       self.model.to(device)
       # Set model to evaluation mode (disables dropout, batch normalization updates)
       self.model.eval()  # Set to evaluation mode (no training)
       
       # Initialize spaCy model for entity recognition
       # This will be used to extract structured medical information from text
       self.nlp = self._load_ner_model()
       
       # Results storage directory
       # Create directory to save evaluation results and model outputs
       self.results_dir = "cim_base_pretrained_results"
       os.makedirs(self.results_dir, exist_ok=True)
       
       # Evaluation metrics storage
       # Dictionary to store computed metrics after evaluation
       self.evaluation_results = {}
       
   def _load_ner_model(self):
       """
       Load SciSpaCy model for named entity recognition in medical texts
       
       SciSpaCy is specifically designed for biomedical and clinical text processing
       and can identify medical entities like diseases, medications, anatomy, etc.
       
       Returns:
           spacy.Language: Loaded SciSpaCy model or None if loading fails
       """
       try:
           # Load the small English scientific model from SciSpaCy
           nlp = spacy.load("en_core_sci_sm")
           print(f"Loaded SciSpaCy model: en_core_sci_sm")
           return nlp
       except (OSError, IOError):
           # Handle case where SciSpaCy model is not installed
           print("Error: SciSpaCy model not available")
           print("Install with: pip install en_core_sci_sm")
           return None
   
   def load_and_preprocess_data(self, sample_size=None):
       """
       Load and preprocess the medical conversation dataset from HuggingFace
       
       This function loads the augmented clinical notes dataset and preprocesses it
       for training and evaluation. The dataset contains medical conversations and
       their corresponding summaries.
       
       Args:
           sample_size (int, optional): Number of samples to use for faster testing
           
       Returns:
           list: List of dictionaries containing conversation-summary pairs
       """
       print("Loading medical dataset...")
       
       try:
           # Load the dataset from HuggingFace hub
           # This dataset contains augmented clinical notes with conversations and summaries
           dataset = load_dataset("AGBonnet/augmented-clinical-notes")
           df = pd.DataFrame(dataset['train'])
           print(f"Loaded {len(df)} samples from HuggingFace dataset")
           
           # Using sample for faster evaluation if specified
           # This is useful during development and testing to reduce computation time
           if sample_size:
               df = df.sample(n=min(sample_size, len(df)), random_state=42)
               print(f"Using sample of {len(df)} records for faster evaluation")
               
       except Exception as e:
           print(f"Error loading dataset: {e}")
           return []
       
       # Preprocess the data
       # Clean and structure the data for model input
       processed_data = []
       for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
           # Extract conversation and summary from each row
           # These are the input-output pairs for our summarization task
           conversation = str(row.get('conversation', ''))
           summary = str(row.get('summary', ''))
           
           # Skip empty entries
           # Filter out invalid samples that don't have sufficient content
           if not conversation or not summary or len(conversation.strip()) < 10:
               continue
               
           # Store valid conversation-summary pairs
           processed_data.append({
               'conversation': conversation,
               'summary': summary
           })
       
       print(f"Processed {len(processed_data)} valid samples")
       return processed_data
   
   def split_data(self, data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
       """
       Split data into train, dev, and test sets (80/10/10)
       
       Standard machine learning practice to have separate datasets for:
       - Training: 80% - used for model training (not used in this baseline)
       - Development: 10% - used for hyperparameter tuning and model selection
       - Test: 10% - used for final evaluation and reporting results
       
       Args:
           data (list): List of data samples
           train_ratio (float): Proportion for training set
           dev_ratio (float): Proportion for development set  
           test_ratio (float): Proportion for test set
           
       Returns:
           tuple: (train_data, dev_data, test_data)
       """
       # Ensure ratios sum to 1.0 (100%)
       assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
       
       # First split: separate train from (dev + test)
       # Use stratified split to maintain data distribution
       train_data, temp_data = train_test_split(
           data, 
           test_size=(dev_ratio + test_ratio), 
           random_state=42  # Fixed seed for reproducibility
       )
       
       # Second split: separate dev from test
       # Calculate relative proportion of dev within the remaining data
       dev_size = dev_ratio / (dev_ratio + test_ratio)
       dev_data, test_data = train_test_split(
           temp_data, 
           test_size=(1 - dev_size), 
           random_state=42  # Fixed seed for reproducibility
       )
       
       # Print split statistics for verification
       print(f"Data Split (80/10/10):")
       print(f"  Train: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
       print(f"  Dev: {len(dev_data)} samples ({len(dev_data)/len(data)*100:.1f}%)")
       print(f"  Test: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
       
       return train_data, dev_data, test_data
   
   def generate_summary(self, text, max_length=128, num_beams=4):
       """
       Generate medical summary using pretrained T5-Base model
       
       This function uses the T5 model's text-to-text capabilities to generate
       summaries from medical conversations. T5 is trained on the prefix "summarize:"
       to indicate the summarization task.
       
       Args:
           text (str): Input medical conversation text
           max_length (int): Maximum length of generated summary
           num_beams (int): Number of beams for beam search decoding
           
       Returns:
           str: Generated summary text
       """
       # Ensure model is in evaluation mode (no gradient computation)
       self.model.eval()
       # Prepend task prefix that T5 was trained on for summarization
       input_text = f"summarize: {text}"
       
       # Tokenize input text
       # Convert text to token IDs that the model can process
       inputs = self.tokenizer.encode(
           input_text,
           return_tensors='pt',  # Return PyTorch tensors
           max_length=512,       # Limit input length to model's maximum
           truncation=True       # Truncate if text is too long
       ).to(device)  # Move to GPU if available
       
       # Generate summary using pretrained T5-Base model
       # Use beam search for better quality generation
       with torch.no_grad():  # Disable gradient computation for efficiency
           summary_ids = self.model.generate(
               inputs,
               max_length=max_length,    # Maximum summary length
               num_beams=num_beams,      # Beam search for better quality
               early_stopping=True,     # Stop when end token is generated
               no_repeat_ngram_size=2   # Prevent repetitive phrases
           )
       
       # Decode summary from token IDs back to text
       summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
       return summary
   
   def extract_medical_entities(self, text):
       """
       Extract entities using spaCy and structure them like the dataset JSON format
       
       This function uses SciSpaCy to identify medical entities in text and structures
       them according to the clinical notes format used in the dataset. This creates
       a structured representation of medical information for better analysis.
       
       Args:
           text (str): Input medical text
           
       Returns:
           dict: Structured medical entities following dataset format
       """
       # Return empty structure if NER model is not available
       if self.nlp is None:
           return {}
       
       # Process text with SciSpaCy to identify medical entities
       doc = self.nlp(text)
       
       # Initialize structured format matching dataset JSON
       # This structure mirrors the format used in the clinical notes dataset
       # Each field represents a different aspect of medical documentation
       entities = {
           "visit motivation": "None",  # Reason for medical visit
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
           "symptoms": [],  # Will be populated with identified symptoms
           "medical examinations": [{"name": "None", "result": "None", "details": "None"}],
           "diagnosis tests": [{"test": "None", "severity": "None", "result": "None", "condition": "None", "time": "None", "details": "None"}],
           "treatments": [],  # Will be populated with identified treatments
           "discharge": {"reason": "None", "referral": "None", "follow up": "None", "discharge summary": "None"}
       }
       
       # Extract entities and populate structured format
       # Initialize lists to collect symptoms and treatments
       symptoms_list = []
       treatments_list = []
       
       # Process each named entity identified by SciSpaCy
       for ent in doc.ents:
           entity_text = ent.text.strip()
           entity_label = ent.label_
           
           # Map spaCy entities to medical structure
           # Use heuristics to categorize entities based on their labels and context
           
           # Identify age-related information
           if entity_label in ['CARDINAL'] and any(age_word in entity_text.lower() for age_word in ['year', 'old', 'age']):
               entities["patient information"]["age"] = entity_text
           
           # Identify gender information
           elif entity_label in ['PERSON'] and any(gender in entity_text.lower() for gender in ['male', 'female', 'man', 'woman']):
               entities["patient information"]["sex"] = entity_text
           
           # Identify medication dosages
           elif entity_label in ['QUANTITY', 'CARDINAL'] and any(unit in entity_text.lower() for unit in ['mg', 'ml', 'dose', 'tablet']):
               # Create treatment entry with dosage information
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
           
           # Identify medications or medical products
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
               # This captures general medical entities that might be symptoms
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
       # Populate symptoms list or use default empty structure
       if symptoms_list:
           entities["symptoms"] = symptoms_list
       else:
           entities["symptoms"] = [{"name of symptom": "None", "intensity of symptom": "None", "location": "None", "time": "None", "temporalisation": "None", "behaviours affecting the symptom": "None", "details": "None"}]
       
       # Populate treatments list or use default empty structure
       if treatments_list:
           entities["treatments"] = treatments_list
       else:
           entities["treatments"] = [{"name": "None", "related condition": "None", "dosage": "None", "time": "None", "frequency": "None", "duration": "None", "reason for taking": "None", "reaction to treatment": "None", "details": "None"}]
       
       return entities
   
   def evaluate_on_test_set(self, test_data):
       """
       Evaluate pretrained T5-Base model on test set: conversation -> summary
       
       This function performs the core evaluation of the baseline model by:
       1. Generating summaries for all test conversations
       2. Comparing generated summaries with reference summaries
       3. Computing multiple evaluation metrics (ROUGE, BERTScore)
       4. Saving results for analysis
       
       Args:
           test_data (list): List of test samples with conversations and summaries
           
       Returns:
           dict: Dictionary containing evaluation metrics and results
       """
       print("="*60)
       print("EVALUATING PRETRAINED T5-BASE MODEL")
       print("Conversation -> Summary Evaluation")
       print("="*60)
       
       # Generate predictions
       # Lists to store model predictions and reference summaries
       predictions = []
       references = []
       
       print(f"Generating predictions for {len(test_data)} samples...")
       
       # Generate summary for each test sample
       for item in tqdm(test_data, desc="Evaluating Pretrained T5-Base"):
           # Generate summary from conversation using pretrained T5-Base
           predicted_summary = self.generate_summary(item['conversation'])
           predictions.append(predicted_summary)
           references.append(item['summary'])  # Reference summary from dataset
       
       # Calculate metrics
       # Compute various evaluation metrics to assess model performance
       results = self._calculate_metrics(predictions, references)
       results['test_samples'] = len(test_data)
       results['model_name'] = self.model_name
       results['model_type'] = 'pretrained'
       
       # Store results for later access
       self.evaluation_results = results
       
       # Save results to files
       # Save detailed results and metrics for further analysis
       self._save_evaluation_results(test_data, predictions, references)
       
       return results
   
   def _calculate_metrics(self, predictions, references):
       """
       Calculate evaluation metrics for model performance assessment
       
       This function computes standard text generation metrics:
       - ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Measures n-gram overlap
       - BERTScore: Measures semantic similarity using BERT embeddings
       - Length statistics: Analyzes summary length characteristics
       
       Args:
           predictions (list): Generated summaries
           references (list): Reference summaries
           
       Returns:
           dict: Dictionary containing computed metrics
       """
       results = {}
       
       # ROUGE Scores
       # ROUGE measures n-gram overlap between generated and reference summaries
       print("Calculating ROUGE scores...")
       scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
       rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
       
       # Calculate ROUGE scores for each prediction-reference pair
       for pred, ref in zip(predictions, references):
           scores = scorer.score(ref, pred)  # Compare reference to prediction
           for metric in rouge_scores:
               # Store F1 score (harmonic mean of precision and recall)
               rouge_scores[metric].append(scores[metric].fmeasure)
       
       # Compute mean and standard deviation for each ROUGE metric
       for metric in rouge_scores:
           results[f'{metric}_mean'] = np.mean(rouge_scores[metric])
           results[f'{metric}_std'] = np.std(rouge_scores[metric])
       
       # BERTScore
       # BERTScore measures semantic similarity using contextual embeddings
       print("Calculating BERTScore...")
       try:
           # Compute BERTScore for all predictions at once
           P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
           results['bertscore_precision'] = P.mean().item()
           results['bertscore_recall'] = R.mean().item()
           results['bertscore_f1'] = F1.mean().item()
       except Exception as e:
           print(f"BERTScore calculation failed: {e}")
       
       # Summary statistics
       # Analyze length characteristics of generated vs reference summaries
       pred_lengths = [len(p.split()) for p in predictions]
       ref_lengths = [len(r.split()) for r in references]
       
       results['avg_pred_length'] = np.mean(pred_lengths)
       results['avg_ref_length'] = np.mean(ref_lengths)
       results['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths)
       
       return results
   
   def _save_evaluation_results(self, test_data, predictions, references):
       """
       Save evaluation results to files for analysis and reporting
       
       This function saves both detailed results (individual predictions) and
       summary metrics to files for further analysis and comparison with other models.
       
       Args:
           test_data (list): Original test data
           predictions (list): Generated summaries
           references (list): Reference summaries
       """
       # Detailed results
       # Save individual predictions alongside original conversations and references
       detailed_results = []
       for i, item in enumerate(test_data):
           detailed_results.append({
               'conversation': item['conversation'],
               'reference_summary': references[i],
               'predicted_summary': predictions[i]
           })
       
       # Save detailed results as CSV for easy analysis
       df_results = pd.DataFrame(detailed_results)
       results_file = os.path.join(self.results_dir, 'detailed_results.csv')
       df_results.to_csv(results_file, index=False)
       
       # Summary metrics
       # Save computed metrics as JSON for programmatic access
       metrics_file = os.path.join(self.results_dir, 'evaluation_metrics.json')
       
       with open(metrics_file, 'w') as f:
           json.dump(self.evaluation_results, f, indent=2)
       
       print(f"Detailed results saved to {results_file}")
       print(f"Metrics saved to {metrics_file}")
   
   def print_results(self):
       """
       Print comprehensive results in a formatted manner
       
       This function provides a clear summary of the evaluation results,
       including model information, performance metrics, and summary statistics.
       """
       print("\n" + "="*60)
       print("PRETRAINED MEDICAL BASELINE EVALUATION RESULTS")
       print("="*60)
       
       if self.evaluation_results:
           # Model information section
           print("\nMODEL INFORMATION:")
           print(f"  Model: {self.evaluation_results['model_name']}")
           print(f"  Type: {self.evaluation_results['model_type']} (no fine-tuning)")
           print(f"  Test Samples: {self.evaluation_results['test_samples']}")
           
           # ROUGE scores section
           # ROUGE-1: Unigram overlap, ROUGE-2: Bigram overlap, ROUGE-L: Longest common subsequence
           print(f"\n  ROUGE SCORES:")
           print(f"    ROUGE-1: {self.evaluation_results.get('rouge1_mean', 0):.4f} (±{self.evaluation_results.get('rouge1_std', 0):.4f})")
           print(f"    ROUGE-2: {self.evaluation_results.get('rouge2_mean', 0):.4f} (±{self.evaluation_results.get('rouge2_std', 0):.4f})")
           print(f"    ROUGE-L: {self.evaluation_results.get('rougeL_mean', 0):.4f} (±{self.evaluation_results.get('rougeL_std', 0):.4f})")
           
           # BERTScore section
           # Semantic similarity metrics using BERT embeddings
           print(f"\n  BERTSCORE:")
           print(f"    Precision: {self.evaluation_results.get('bertscore_precision', 0):.4f}")
           print(f"    Recall: {self.evaluation_results.get('bertscore_recall', 0):.4f}")
           print(f"    F1: {self.evaluation_results.get('bertscore_f1', 0):.4f}")
           
           # Summary statistics section
           # Length analysis of generated vs reference summaries
           print(f"\n  SUMMARY STATISTICS:")
           print(f"    Avg Predicted Length: {self.evaluation_results['avg_pred_length']:.1f} words")
           print(f"    Avg Reference Length: {self.evaluation_results['avg_ref_length']:.1f} words")
           print(f"    Length Ratio: {self.evaluation_results['length_ratio']:.3f}")
       
       print(f"\nResults saved to: {self.results_dir}")

def main():
   """
   Run Medical Baseline evaluation with pretrained model
   
   This is the main execution function that orchestrates the entire evaluation process:
   1. Initialize the baseline model
   2. Load and preprocess the dataset
   3. Split data into train/dev/test sets
   4. Evaluate the pretrained model on the test set
   5. Display comprehensive results
   
   Returns:
       CIM_Baseline_Pretrained: The initialized and evaluated baseline model
   """
   print("="*60)
   print("MEDICAL BASELINE")
   print("Pretrained T5-Base + spaCy (80/10/10 Split)")
   print("Conversation -> Summary Evaluation")
   print("No Fine-tuning - Using Pretrained Model Only")
   print("="*60)
   
   # Initialize baseline
   # Create an instance of the baseline model with pretrained components
   baseline = CIM_Baseline_Pretrained()
   
   # Load and split data
   # Load the medical dataset and split it into train/dev/test sets
   data = baseline.load_and_preprocess_data(sample_size=1000)  # Use only 1000 samples for speed
   if not data:
       print("Error: No data loaded. Exiting.")
       return None
   
   # Split data using standard 80/10/10 ratio
   train_data, dev_data, test_data = baseline.split_data(data)
   
   # No training - directly evaluate on test set using pretrained model
   # This baseline skips training and uses the pretrained T5-Base model as-is
   print("\nSkipping training - using pretrained T5-Base model")
   
   # Evaluate on test set
   # Perform comprehensive evaluation using the test set
   baseline.evaluate_on_test_set(test_data)
   
   # Print comprehensive results
   # Display formatted results including all metrics and statistics
   baseline.print_results()
   
   return baseline

# Execute the main function when script is run directly
if __name__ == "__main__":
   baseline = main()