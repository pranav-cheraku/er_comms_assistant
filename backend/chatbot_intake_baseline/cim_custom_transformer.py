# backend/chatbot_intake_baseline/cim_custom_transformer.py

"""
Enhanced Custom Medical Transformer
Updated to follow the specified architecture requirements
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import spacy
import json
from tqdm import tqdm
import os
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnhancedMedicalTransformer(nn.Module):
    """Enhanced transformer for medical summarization following exact architecture specifications"""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=4, 
                 d_ff=2048, max_seq_length=512, max_target_length=128, dropout=0.2):
        super(EnhancedMedicalTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Shared embeddings with dropout
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Encoder with specified architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                    # 512
            nhead=num_heads,                    # 8
            dim_feedforward=d_ff,               # 2048 (updated from 1024)
            dropout=dropout,                    # 0.2
            activation='gelu',                  # GELU activation
            batch_first=True,
            norm_first=True                     # Pre-LayerNorm for better gradient flow
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 4 blocks
        
        # Decoder with specified architecture
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,                    # 512
            nhead=num_heads,                    # 8
            dim_feedforward=d_ff,               # 2048 (updated from 1024)
            dropout=dropout,                    # 0.2
            activation='gelu',                  # GELU activation
            batch_first=True,
            norm_first=True                     # Pre-LayerNorm for better gradient flow
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)  # 4 blocks
        
        # Output projection with dropout and layer norm
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Positional encoding - ensuring max_len >= 2048 as specified
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Additional regularization (applied at multiple levels)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding with max_len >= 2048"""
        max_len = max(max_len, 2048)  # Ensure max_len >= 2048 as specified
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)
    
    def _init_weights(self):
        """Improved weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def encode(self, src, src_mask=None):
        """Encode source sequence with regularization"""
        # Add positional encoding
        seq_len = src.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :].to(src.device)
        
        # Embed and encode with dropout (applied at multiple levels)
        src_emb = self.shared_embedding(src) * np.sqrt(self.d_model)
        src_emb = src_emb + pos_enc
        src_emb = self.embedding_dropout(src_emb)
        
        # Create padding mask
        if src_mask is not None:
            src_key_padding_mask = (src_mask == 0)
        else:
            src_key_padding_mask = None
        
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Decode target sequence with regularization"""
        seq_len = tgt.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :].to(tgt.device)
        
        # Embed target with dropout (applied at multiple levels)
        tgt_emb = self.shared_embedding(tgt) * np.sqrt(self.d_model)
        tgt_emb = tgt_emb + pos_enc
        tgt_emb = self.embedding_dropout(tgt_emb)
        
        # Create causal mask for target
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        # Create key padding masks
        tgt_key_padding_mask = None
        memory_key_padding_mask = None
        if memory_mask is not None:
            memory_key_padding_mask = (memory_mask == 0)
        
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Apply output normalization and dropout (applied at multiple levels)
        output = self.output_norm(output)
        output = self.output_dropout(output)
        
        return self.output_projection(output)
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt=None, src_mask=None, memory_mask=None):
        """Forward pass"""
        memory = self.encode(src, src_mask)
        
        if tgt is not None:
            # Training mode
            output = self.decode(tgt, memory, memory_mask=src_mask)
            return output
        else:
            # Inference mode
            return memory

class MedicalDataset(Dataset):
    """Dataset for medical conversation summarization"""
    
    def __init__(self, tokenizer, data, max_source_length=512, max_target_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        conversation = str(item.get('conversation', ''))
        summary = str(item.get('summary', ''))
        
        # Tokenize source
        source_encoding = self.tokenizer(
            conversation,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target - T5 format expects no special prefix
        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'target_ids': target_encoding['input_ids'].flatten(),
            'conversation': conversation,
            'summary': summary
        }

class CIM_Enhanced_Transformer:
    """
    Enhanced Custom Transformer for Medical Summarization
    Updated to follow exact architecture specifications:
    - Embedding Dim: 512
    - Number of Attention Heads: 8  
    - Number of Blocks: 4 encoder + 4 decoder
    - Feedforward Size: 2048
    - Dropout: 0.2 (applied at multiple levels)
    - Positional Encoding: Sinusoidal (max_len >= 2048)
    - Normalization: Pre-LayerNorm for better gradient flow
    - Activation: GELU (smooth, alternative to ReLU)
    - Vocab Size: Based on T5Tokenizer
    """
    
    def __init__(self, model_name='t5-small'):
        # Use T5 tokenizer for proper seq2seq tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        
        # Initialize enhanced custom transformer with exact architecture specifications
        self.model = EnhancedMedicalTransformer(
            vocab_size=self.vocab_size,
            d_model=512,                    # Embedding Dim
            num_heads=8,                    # Number of Attention Heads
            num_layers=4,                   # Number of Blocks (4 encoder + 4 decoder)
            d_ff=2048,                      # Feedforward Size (updated from 1024)
            max_seq_length=512,
            max_target_length=128,
            dropout=0.2                     # Dropout (applied at multiple levels)
        ).to(device)
        
        # Initialize spaCy
        self.nlp = self._load_ner_model()
        
        # Results storage
        self.results_dir = "4cim_custom_transformer_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.training_results = {}
        self.evaluation_results = {}
        
        print(f"Initialized Enhanced Custom Medical Transformer")
        print(f"Architecture Summary:")
        print(f"  - Embedding Dim: {512}")
        print(f"  - Attention Heads: {8}")
        print(f"  - Encoder Blocks: {4}")
        print(f"  - Decoder Blocks: {4}")
        print(f"  - Feedforward Size: {2048}")
        print(f"  - Dropout: {0.2} (applied at multiple levels)")
        print(f"  - Positional Encoding: Sinusoidal (max_len >= 2048)")
        print(f"  - Normalization: Pre-LayerNorm")
        print(f"  - Activation: GELU")
        print(f"  - Vocab Size: {self.vocab_size} (T5Tokenizer)")
        print(f"  - Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_ner_model(self):
        """Load SciSpaCy model"""
        try:
            nlp = spacy.load("en_core_sci_sm")
            return nlp
        except (OSError, IOError):
            return None
    
    def load_and_preprocess_data(self, sample_size=None):
        """Load data with minimal output"""
        try:
            dataset = load_dataset("AGBonnet/augmented-clinical-notes")
            df = pd.DataFrame(dataset['train'])
            
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
                
        except Exception as e:
            return []
        
        # Process data
        processed_data = []
        for idx, row in df.iterrows():
            conversation = str(row.get('conversation', ''))
            summary = str(row.get('summary', ''))
            
            if conversation and summary and len(conversation.strip()) > 10:
                processed_data.append({
                    'conversation': conversation,
                    'summary': summary
                })
        
        return processed_data
    
    def split_data(self, data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
        """Split data"""
        train_data, temp_data = train_test_split(
            data, test_size=(dev_ratio + test_ratio), random_state=42
        )
        
        dev_size = dev_ratio / (dev_ratio + test_ratio)
        dev_data, test_data = train_test_split(
            temp_data, test_size=(1 - dev_size), random_state=42
        )
        
        return train_data, dev_data, test_data
    
    def create_datasets(self, train_data, dev_data, test_data):
        """Create datasets"""
        train_dataset = MedicalDataset(self.tokenizer, train_data)
        dev_dataset = MedicalDataset(self.tokenizer, dev_data)
        test_dataset = MedicalDataset(self.tokenizer, test_data)
        
        return train_dataset, dev_dataset, test_dataset
    
    def train_model(self, train_dataset, dev_dataset, num_epochs=7, batch_size=12, learning_rate=1e-4):
        """Enhanced training with better optimization and regularization"""
        print("="*60)
        print("TRAINING ENHANCED CUSTOM MEDICAL TRANSFORMER")
        print("="*60)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        
        # Enhanced optimizer with better settings
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        total_steps = len(train_loader) * num_epochs
        
        # Enhanced learning rate scheduling
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=total_steps // 20,
            num_training_steps=total_steps
        )
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=0.1
        )
        
        # Training tracking
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        train_losses = []
        val_losses = []
        
        print(f"Training for up to {num_epochs} epochs with early stopping")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc="Training", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                # Prepare decoder input (shift right)
                decoder_input_ids = target_ids[:, :-1].contiguous()
                labels = target_ids[:, 1:].contiguous()
                
                # Forward pass
                outputs = self.model(input_ids, decoder_input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(outputs.view(-1, self.vocab_size), labels.view(-1))
                
                # Add L2 regularization manually
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2) ** 2
                loss += 1e-6 * l2_reg
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (important for stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self._validate(dev_loader, criterion)
            val_losses.append(val_loss)
            
            # Learning rate logging
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                model_path = os.path.join(self.results_dir, 'best_model.pt')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, model_path)
                print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"  Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Save training results
        self.training_results = {
            'final_train_loss': avg_train_loss,
            'best_val_loss': best_val_loss,
            'train_samples': len(train_dataset),
            'dev_samples': len(dev_dataset),
            'epochs_completed': epoch + 1,
            'model_name': 'Enhanced Custom Medical Transformer',
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"Training completed after {epoch + 1} epochs!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def _validate(self, dev_loader, criterion):
        """Enhanced validation"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                decoder_input_ids = target_ids[:, :-1].contiguous()
                labels = target_ids[:, 1:].contiguous()
                
                outputs = self.model(input_ids, decoder_input_ids, attention_mask)
                loss = criterion(outputs.view(-1, self.vocab_size), labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def load_best_model(self):
        """Load best model with full checkpoint"""
        model_path = os.path.join(self.results_dir, 'best_model.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print("No saved model found")
    
    def debug_model_output(self, sample_text):
        """Debug what the model is actually learning"""
        print("="*50)
        print("MODEL DEBUGGING")
        print("="*50)
        
        inputs = self.tokenizer(sample_text[:200], return_tensors='pt', max_length=512, truncation=True, padding=True).to(device)
        
        self.model.eval()
        with torch.no_grad():
            # Check encoder output
            memory = self.model.encode(inputs['input_ids'], inputs['attention_mask'])
            print(f"Encoder output shape: {memory.shape}")
            print(f"Encoder output mean: {memory.mean():.4f}")
            print(f"Encoder output std: {memory.std():.4f}")
            
            # Check decoder with teacher forcing
            dummy_target = torch.zeros(1, 10).long().to(device)
            decoder_out = self.model.decode(dummy_target, memory)
            print(f"Decoder output shape: {decoder_out.shape}")
            print(f"Decoder logits mean: {decoder_out.mean():.4f}")
            print(f"Decoder logits std: {decoder_out.std():.4f}")
            
            # Check vocabulary distribution
            probs = F.softmax(decoder_out[0, -1, :], dim=-1)
            top_tokens = torch.topk(probs, 10)
            print(f"Top 10 token probabilities:")
            for i, (prob, token_id) in enumerate(zip(top_tokens[0], top_tokens[1])):
                token = self.tokenizer.decode([token_id.item()])
                print(f"  {i+1}. {token} ({prob:.4f})")
        
        print("="*50)
    
    def generate_summary(self, text, max_length=100):
        """Enhanced generation with nucleus sampling and temperature"""
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            # Encode
            memory = self.model.encode(inputs['input_ids'], inputs['attention_mask'])
            
            # Start generation with proper T5 decoder start token
            generated = [self.tokenizer.pad_token_id]
            
            for step in range(max_length):
                # Prepare decoder input
                decoder_input = torch.tensor([generated]).to(device)
                
                # Ensure we don't exceed positional encoding limits (2048)
                if decoder_input.size(1) >= 2048:
                    break
                
                # Decode
                output = self.model.decode(decoder_input, memory, memory_mask=inputs['attention_mask'])
                
                # Get next token probabilities
                next_token_logits = output[0, -1, :]
                
                # Apply temperature for controlled randomness
                temperature = 0.7
                next_token_logits = next_token_logits / temperature
                
                # Use nucleus (top-p) sampling for better quality
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Top-p (nucleus) sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold (nucleus)
                top_p = 0.9
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Prevent generating pad tokens in the middle
                if next_token == self.tokenizer.pad_token_id and len(generated) > 1:
                    break
                
                generated.append(next_token)
                
                # Stop conditions
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            # Decode to text, skip the initial pad token
            if len(generated) > 1:
                summary = self.tokenizer.decode(generated[1:], skip_special_tokens=True)
            else:
                summary = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            # Clean up and post-process
            summary = summary.strip()
            
            # Remove any repetitive patterns (simple deduplication)
            words = summary.split()
            if len(words) > 3:
                # Remove immediate repetitions
                cleaned_words = [words[0]]
                for i in range(1, len(words)):
                    if words[i] != words[i-1]:
                        cleaned_words.append(words[i])
                summary = " ".join(cleaned_words)
            
            if not summary:
                summary = "No summary generated"
            
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
        """Evaluate with minimal output"""
        print("="*60)
        print("EVALUATING ENHANCED CUSTOM MEDICAL TRANSFORMER")
        print("="*60)
        
        self.load_best_model()
        
        predictions = []
        references = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            predicted_summary = self.generate_summary(item['conversation'])
            predictions.append(predicted_summary)
            references.append(item['summary'])
        
        # Calculate metrics
        results = self._calculate_metrics(predictions, references)
        results['test_samples'] = len(test_data)
        results['model_name'] = 'Enhanced Custom Medical Transformer'
        
        self.evaluation_results = results
        
        # Save results
        self._save_evaluation_results(test_data, predictions, references)
        
        return results
    
    def _calculate_metrics(self, predictions, references):
        """Calculate metrics"""
        results = {}
        
        # ROUGE Scores
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
        try:
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
            results['bertscore_precision'] = P.mean().item()
            results['bertscore_recall'] = R.mean().item()
            results['bertscore_f1'] = F1.mean().item()
        except Exception as e:
            results['bertscore_precision'] = 0.0
            results['bertscore_recall'] = 0.0
            results['bertscore_f1'] = 0.0
        
        # Summary statistics
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        results['avg_pred_length'] = np.mean(pred_lengths)
        results['avg_ref_length'] = np.mean(ref_lengths)
        results['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths)
        
        return results
    
    def _save_evaluation_results(self, test_data, predictions, references):
        """Save results"""
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
        
        metrics_file = os.path.join(self.results_dir, 'evaluation_metrics.json')
        combined_results = {
            'training_results': self.training_results,
            'evaluation_results': self.evaluation_results
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
    
    def print_results(self):
        """Print results (minimal output)"""
        print("\n" + "="*60)
        print("ENHANCED CUSTOM MEDICAL TRANSFORMER RESULTS")
        print("="*60)
        
        if self.evaluation_results:
            print(f"\nMODEL INFORMATION:")
            print(f"  Model: {self.evaluation_results['model_name']}")
            print(f"  Type: enhanced custom transformer (fine-tuned)")
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

def main():
    """Main execution with enhanced transformer - minimal terminal output"""
    transformer = CIM_Enhanced_Transformer()
    
    # Load data
    data = transformer.load_and_preprocess_data(sample_size=15000)
    if not data:
        return None
    
    train_data, dev_data, test_data = transformer.split_data(data)
    train_dataset, dev_dataset, test_dataset = transformer.create_datasets(train_data, dev_data, test_data)
    
    # Train with enhanced settings
    transformer.train_model(
        train_dataset, dev_dataset,
        num_epochs=25, batch_size=12, learning_rate=1e-4
    )
    
    # Evaluate
    transformer.evaluate_on_test_set(test_data)
    
    # Print results
    transformer.print_results()
    
    return transformer

if __name__ == "__main__":
    transformer = main()