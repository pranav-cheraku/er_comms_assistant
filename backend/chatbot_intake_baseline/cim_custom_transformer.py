# backend/chatbot_intake_baseline/cim_custom_transformer.py

"""
Enhanced Custom Medical Transformer
Updated to follow the specified architecture requirements

This module implements a custom transformer architecture specifically designed for medical conversation
summarization. The transformer follows an encoder-decoder architecture with the following specifications:
- Embedding Dimension: 512
- Number of Attention Heads: 8
- Number of Blocks: 4 encoder + 4 decoder layers
- Feedforward Size: 2048
- Dropout: 0.2 (applied at multiple levels for regularization)
- Positional Encoding: Sinusoidal with max_len >= 2048
- Normalization: Pre-LayerNorm for better gradient flow
- Activation: GELU (smooth alternative to ReLU)
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

# Device configuration - utilize GPU if available for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnhancedMedicalTransformer(nn.Module):
    """
    Enhanced transformer for medical summarization following exact architecture specifications
    
    This class implements a complete encoder-decoder transformer architecture optimized for
    medical text summarization. The model uses shared embeddings between encoder and decoder
    and implements various regularization techniques for stable training.
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=4, 
                 d_ff=2048, max_seq_length=512, max_target_length=128, dropout=0.2):
        """
        Initialize the Enhanced Medical Transformer
        
        Args:
            vocab_size (int): Size of the vocabulary (determined by tokenizer)
            d_model (int): Embedding dimension (512 as specified)
            num_heads (int): Number of attention heads (8 as specified)
            num_layers (int): Number of transformer blocks (4 for both encoder and decoder)
            d_ff (int): Feedforward network dimension (2048 as specified)
            max_seq_length (int): Maximum input sequence length
            max_target_length (int): Maximum target sequence length
            dropout (float): Dropout rate for regularization (0.2 as specified)
        """
        super(EnhancedMedicalTransformer, self).__init__()
        
        # Store key dimensions for later use
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Shared embeddings between encoder and decoder to reduce parameters and improve learning
        # This technique allows the model to learn consistent representations across input and output
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)  # Dropout after embeddings for regularization
        
        # Encoder architecture following specifications:
        # - 4 transformer blocks
        # - 8 attention heads per block
        # - 2048 feedforward dimension
        # - Pre-LayerNorm for better gradient flow and training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                    # 512 - embedding dimension
            nhead=num_heads,                    # 8 - number of attention heads
            dim_feedforward=d_ff,               # 2048 - feedforward network size
            dropout=dropout,                    # 0.2 - dropout for regularization
            activation='gelu',                  # GELU activation (smoother than ReLU)
            batch_first=True,                   # Batch dimension comes first
            norm_first=True                     # Pre-LayerNorm for better gradient flow
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # Stack 4 blocks
        
        # Decoder architecture matching encoder specifications:
        # - Same 4 transformer blocks structure
        # - Includes cross-attention to encoder outputs
        # - Causal self-attention for autoregressive generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,                    # 512 - embedding dimension
            nhead=num_heads,                    # 8 - number of attention heads
            dim_feedforward=d_ff,               # 2048 - feedforward network size
            dropout=dropout,                    # 0.2 - dropout for regularization
            activation='gelu',                  # GELU activation function
            batch_first=True,                   # Batch dimension comes first
            norm_first=True                     # Pre-LayerNorm architecture
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)  # Stack 4 blocks
        
        # Output projection layers with additional regularization
        # These layers convert the final hidden states to vocabulary logits
        self.output_norm = nn.LayerNorm(d_model)        # Layer normalization before output
        self.output_dropout = nn.Dropout(dropout)       # Final dropout layer
        self.output_projection = nn.Linear(d_model, vocab_size)  # Project to vocabulary size
        
        # Positional encoding - using sinusoidal encoding as specified
        # Ensures max_len >= 2048 to handle long sequences effectively
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Additional regularization dropout (applied at multiple levels as specified)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize all model weights using appropriate techniques
        self._init_weights()
    
    def _create_positional_encoding(self, max_len, d_model):
        """
        Create sinusoidal positional encoding with max_len >= 2048
        
        Positional encoding adds information about token positions to embeddings since
        transformers don't inherently understand sequence order. We use sinusoidal
        encoding which allows the model to attend to relative positions.
        
        Args:
            max_len (int): Maximum sequence length to encode
            d_model (int): Model dimension for encoding
            
        Returns:
            torch.Tensor: Positional encoding matrix of shape (1, max_len, d_model)
        """
        max_len = max(max_len, 2048)  # Ensure max_len >= 2048 as specified in requirements
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the sinusoidal pattern using different frequencies
        # This allows the model to learn relative positions effectively
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        # Apply sine to even dimensions and cosine to odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def _init_weights(self):
        """
        Improved weight initialization using Xavier/Glorot initialization
        
        Proper weight initialization is crucial for stable training and convergence.
        We use different initialization strategies for different layer types:
        - Xavier normal for linear layers (good for tanh/sigmoid activations)
        - Normal distribution for embeddings
        - Constant initialization for layer norm parameters
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                # This helps maintain activation magnitudes across layers
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                # Small random initialization for embeddings
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                # Standard initialization for layer normalization
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence with regularization
        
        The encoder processes the input sequence (medical conversation) and creates
        contextual representations that capture the meaning and relationships
        between different parts of the conversation.
        
        Args:
            src (torch.Tensor): Source token indices of shape (batch_size, seq_len)
            src_mask (torch.Tensor, optional): Attention mask for padding tokens
            
        Returns:
            torch.Tensor: Encoded representations of shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding to embeddings
        seq_len = src.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :].to(src.device)
        
        # Embed tokens and scale by sqrt(d_model) as in original transformer paper
        # This scaling helps balance the magnitude of embeddings and positional encoding
        src_emb = self.shared_embedding(src) * np.sqrt(self.d_model)
        src_emb = src_emb + pos_enc
        src_emb = self.embedding_dropout(src_emb)  # Apply dropout for regularization
        
        # Create padding mask to ignore padding tokens during attention
        if src_mask is not None:
            src_key_padding_mask = (src_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Pass through encoder layers
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Decode target sequence with regularization
        
        The decoder generates the summary autoregressively, using both the
        encoded input (via cross-attention) and previously generated tokens
        (via causal self-attention).
        
        Args:
            tgt (torch.Tensor): Target token indices of shape (batch_size, tgt_len)
            memory (torch.Tensor): Encoder outputs of shape (batch_size, src_len, d_model)
            tgt_mask (torch.Tensor, optional): Causal mask for target sequence
            memory_mask (torch.Tensor, optional): Mask for encoder outputs
            
        Returns:
            torch.Tensor: Logits over vocabulary of shape (batch_size, tgt_len, vocab_size)
        """
        seq_len = tgt.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :].to(tgt.device)
        
        # Embed target tokens with positional encoding
        tgt_emb = self.shared_embedding(tgt) * np.sqrt(self.d_model)
        tgt_emb = tgt_emb + pos_enc
        tgt_emb = self.embedding_dropout(tgt_emb)  # Apply dropout for regularization
        
        # Create causal mask to prevent the model from seeing future tokens
        # This ensures autoregressive generation during training
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        # Create key padding masks for attention mechanisms
        tgt_key_padding_mask = None
        memory_key_padding_mask = None
        if memory_mask is not None:
            memory_key_padding_mask = (memory_mask == 0)
        
        # Pass through decoder layers with cross-attention to encoder
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,                              # Causal mask for self-attention
            tgt_key_padding_mask=tgt_key_padding_mask,      # Padding mask for target
            memory_key_padding_mask=memory_key_padding_mask  # Padding mask for encoder outputs
        )
        
        # Apply final normalization and dropout before output projection
        output = self.output_norm(output)
        output = self.output_dropout(output)
        
        # Project to vocabulary size to get logits for each token
        return self.output_projection(output)
    
    def _generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for autoregressive generation
        
        This mask ensures that when predicting token i, the model can only
        attend to tokens 0 through i-1, maintaining the autoregressive property.
        
        Args:
            sz (int): Size of the square mask
            
        Returns:
            torch.Tensor: Upper triangular mask with -inf above diagonal
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt=None, src_mask=None, memory_mask=None):
        """
        Forward pass of the transformer
        
        This method handles both training (when target is provided) and
        inference modes (when only source is provided).
        
        Args:
            src (torch.Tensor): Source sequence token indices
            tgt (torch.Tensor, optional): Target sequence for training
            src_mask (torch.Tensor, optional): Source padding mask
            memory_mask (torch.Tensor, optional): Memory padding mask
            
        Returns:
            torch.Tensor: Either logits (training) or encoded memory (inference)
        """
        # Always encode the source sequence
        memory = self.encode(src, src_mask)
        
        if tgt is not None:
            # Training mode: return logits for loss calculation
            output = self.decode(tgt, memory, memory_mask=src_mask)
            return output
        else:
            # Inference mode: return encoded memory for generation
            return memory

class MedicalDataset(Dataset):
    """
    Dataset class for medical conversation summarization
    
    This class handles the preprocessing and tokenization of medical conversation
    data, preparing it for training the transformer model. It tokenizes both
    the input conversations and target summaries according to the model's requirements.
    """
    
    def __init__(self, tokenizer, data, max_source_length=512, max_target_length=128):
        """
        Initialize the medical dataset
        
        Args:
            tokenizer: Tokenizer for text preprocessing (T5 tokenizer)
            data (list): List of conversation-summary pairs
            max_source_length (int): Maximum length for source sequences
            max_target_length (int): Maximum length for target sequences
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset
        
        This method tokenizes both the conversation and summary, preparing them
        for model input. It handles padding and truncation to ensure consistent
        sequence lengths within batches.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            dict: Dictionary containing tokenized inputs and targets
        """
        item = self.data[idx]
        
        # Extract conversation and summary, ensuring they are strings
        conversation = str(item.get('conversation', ''))
        summary = str(item.get('summary', ''))
        
        # Tokenize source conversation with padding and truncation
        # This ensures all sequences in a batch have the same length
        source_encoding = self.tokenizer(
            conversation,
            max_length=self.max_source_length,
            padding='max_length',               # Pad to max_length
            truncation=True,                    # Truncate if longer than max_length
            return_tensors='pt'                 # Return PyTorch tensors
        )
        
        # Tokenize target summary - T5 format expects no special prefix
        # The model will learn to generate summaries directly
        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].flatten(),        # Source token IDs
            'attention_mask': source_encoding['attention_mask'].flatten(), # Source attention mask
            'target_ids': target_encoding['input_ids'].flatten(),       # Target token IDs
            'conversation': conversation,                                # Original text for debugging
            'summary': summary                                          # Original summary for evaluation
        }

class CIM_Enhanced_Transformer:
    """
    Enhanced Custom Transformer for Medical Summarization
    
    This is the main class that orchestrates the entire pipeline for medical text
    summarization using a custom transformer architecture. It handles data loading,
    model training, evaluation, and named entity recognition.
    
    Architecture specifications implemented:
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
        """
        Initialize the Enhanced Custom Transformer system
        
        Args:
            model_name (str): Base model name for tokenizer (default: 't5-small')
        """
        # Use T5 tokenizer for proper sequence-to-sequence token handling
        # T5 tokenizer includes special tokens needed for generation tasks
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        
        # Initialize enhanced custom transformer with exact architecture specifications
        self.model = EnhancedMedicalTransformer(
            vocab_size=self.vocab_size,
            d_model=512,                    # Embedding Dim as specified
            num_heads=8,                    # Number of Attention Heads as specified
            num_layers=4,                   # Number of Blocks (4 encoder + 4 decoder)
            d_ff=2048,                      # Feedforward Size as specified
            max_seq_length=512,             # Maximum input sequence length
            max_target_length=128,          # Maximum target sequence length
            dropout=0.2                     # Dropout rate applied at multiple levels
        ).to(device)
        
        # Initialize spaCy for Named Entity Recognition
        # This will be used for extracting medical entities from text
        self.nlp = self._load_ner_model()
        
        # Create results directory for saving outputs
        self.results_dir = "4cim_custom_transformer_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize result storage dictionaries
        self.training_results = {}
        self.evaluation_results = {}
        
        # Print architecture summary for verification
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
        """
        Load SciSpaCy model for Named Entity Recognition
        
        SciSpaCy is specifically designed for biomedical and scientific text,
        making it ideal for extracting medical entities from conversations.
        
        Returns:
            spacy.Language: Loaded spaCy model or None if unavailable
        """
        try:
            nlp = spacy.load("en_core_sci_sm")
            return nlp
        except (OSError, IOError):
            # Handle case where SciSpaCy model is not installed
            print("Warning: SciSpaCy model not found. NER functionality will be limited.")
            return None
    
    def load_and_preprocess_data(self, sample_size=None):
        """
        Load and preprocess the augmented clinical notes dataset
        
        This method loads the medical conversation dataset from HuggingFace,
        performs basic preprocessing, and prepares it for training.
        
        Args:
            sample_size (int, optional): Number of samples to use (None for all)
            
        Returns:
            list: Processed data with conversation-summary pairs
        """
        try:
            # Load the augmented clinical notes dataset from HuggingFace
            # This dataset contains medical conversations and their summaries
            dataset = load_dataset("AGBonnet/augmented-clinical-notes")
            df = pd.DataFrame(dataset['train'])
            
            # Sample data if sample_size is specified (useful for testing/debugging)
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
        
        # Process and filter the data
        processed_data = []
        for idx, row in df.iterrows():
            conversation = str(row.get('conversation', ''))
            summary = str(row.get('summary', ''))
            
            # Filter out invalid or very short conversations
            # This ensures we only train on meaningful data
            if conversation and summary and len(conversation.strip()) > 10:
                processed_data.append({
                    'conversation': conversation,
                    'summary': summary
                })
        
        print(f"Loaded and processed {len(processed_data)} conversation-summary pairs")
        return processed_data
    
    def split_data(self, data, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
        """
        Split data into training, development, and test sets
        
        Proper data splitting is crucial for unbiased evaluation. We use
        stratified splitting to ensure representative samples in each set.
        
        Args:
            data (list): Complete dataset
            train_ratio (float): Proportion for training (default: 0.8)
            dev_ratio (float): Proportion for development/validation (default: 0.1)
            test_ratio (float): Proportion for testing (default: 0.1)
            
        Returns:
            tuple: (train_data, dev_data, test_data)
        """
        # First split: separate training from (dev + test)
        train_data, temp_data = train_test_split(
            data, test_size=(dev_ratio + test_ratio), random_state=42
        )
        
        # Second split: separate dev from test
        dev_size = dev_ratio / (dev_ratio + test_ratio)
        dev_data, test_data = train_test_split(
            temp_data, test_size=(1 - dev_size), random_state=42
        )
        
        print(f"Data split - Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
        return train_data, dev_data, test_data
    
    def create_datasets(self, train_data, dev_data, test_data):
        """
        Create PyTorch datasets from the split data
        
        This method wraps the data splits in our custom MedicalDataset class,
        which handles tokenization and formatting for the model.
        
        Args:
            train_data (list): Training data
            dev_data (list): Development data  
            test_data (list): Test data
            
        Returns:
            tuple: (train_dataset, dev_dataset, test_dataset)
        """
        train_dataset = MedicalDataset(self.tokenizer, train_data)
        dev_dataset = MedicalDataset(self.tokenizer, dev_data)
        test_dataset = MedicalDataset(self.tokenizer, test_data)
        
        return train_dataset, dev_dataset, test_dataset
    
    def train_model(self, train_dataset, dev_dataset, num_epochs=7, batch_size=12, learning_rate=1e-4):
        """
        Enhanced training with better optimization and regularization
        
        This method implements the complete training loop with modern best practices:
        - AdamW optimizer with weight decay for better generalization
        - Learning rate scheduling with warmup
        - Early stopping to prevent overfitting
        - Gradient clipping for training stability
        - Label smoothing for better calibration
        
        Args:
            train_dataset (MedicalDataset): Training dataset
            dev_dataset (MedicalDataset): Development dataset for validation
            num_epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Initial learning rate
        """
        print("="*60)
        print("TRAINING ENHANCED CUSTOM MEDICAL TRANSFORMER")
        print("="*60)
        
        # Create data loaders for batch processing
        # Shuffle training data to improve convergence
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        
        # Enhanced optimizer configuration
        # AdamW includes weight decay (L2 regularization) for better generalization
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,          # L2 regularization
            betas=(0.9, 0.98),          # Beta parameters for Adam
            eps=1e-9                    # Epsilon for numerical stability
        )
        
        # Calculate total training steps for learning rate scheduling
        total_steps = len(train_loader) * num_epochs
        
        # Enhanced learning rate scheduling with warmup
        # Warmup helps stabilize training in the early stages
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=total_steps // 20,    # 5% of steps for warmup
            num_training_steps=total_steps
        )
        
        # Loss function with label smoothing for better calibration
        # Label smoothing prevents the model from being overconfident
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,   # Ignore padding tokens in loss
            label_smoothing=0.1                         # Apply label smoothing
        )
        
        # Early stopping configuration to prevent overfitting
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5                    # Stop if no improvement for 5 epochs
        
        # Track training progress
        train_losses = []
        val_losses = []
        
        print(f"Training for up to {num_epochs} epochs with early stopping")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Main training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()              # Set model to training mode
            total_loss = 0
            num_batches = 0
            
            # Process each batch in the training set
            for batch in tqdm(train_loader, desc="Training", leave=False):
                # Move batch data to device (GPU if available)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                # Prepare decoder input by shifting target sequence right
                # This is the standard approach for teacher forcing in training
                decoder_input_ids = target_ids[:, :-1].contiguous()    # Remove last token
                labels = target_ids[:, 1:].contiguous()                # Remove first token
                
                # Forward pass through the model
                outputs = self.model(input_ids, decoder_input_ids, attention_mask)
                
                # Calculate loss between predictions and true labels
                loss = criterion(outputs.view(-1, self.vocab_size), labels.view(-1))
                
                # Add L2 regularization manually for additional regularization
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2) ** 2
                loss += 1e-6 * l2_reg          # Small L2 penalty
                
                # Backward pass and optimization
                optimizer.zero_grad()           # Clear previous gradients
                loss.backward()                 # Compute gradients
                
                # Gradient clipping for training stability
                # This prevents exploding gradients which can destabilize training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()                # Update parameters
                scheduler.step()                # Update learning rate
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average training loss for this epoch
            avg_train_loss = total_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase to monitor overfitting
            val_loss = self._validate(dev_loader, criterion)
            val_losses.append(val_loss)
            
            # Log current learning rate for monitoring
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
            
            # Early stopping and model saving logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save the best model checkpoint
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
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"  Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Save comprehensive training results for analysis
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
        """
        Enhanced validation function to monitor training progress
        
        This function evaluates the model on the development set without
        updating parameters. It's used during training to monitor overfitting
        and implement early stopping.
        
        Args:
            dev_loader (DataLoader): Development set data loader
            criterion (nn.Module): Loss function for evaluation
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()               # Set model to evaluation mode
        total_loss = 0
        num_batches = 0
        
        # Disable gradient computation for efficiency during validation
        with torch.no_grad():
            for batch in dev_loader:
                # Move batch data to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_ids = batch['target_ids'].to(device)
                
                # Prepare inputs same as training
                decoder_input_ids = target_ids[:, :-1].contiguous()
                labels = target_ids[:, 1:].contiguous()
                
                # Forward pass (no gradient computation)
                outputs = self.model(input_ids, decoder_input_ids, attention_mask)
                loss = criterion(outputs.view(-1, self.vocab_size), labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def load_best_model(self):
        """
        Load the best model checkpoint saved during training
        
        This method loads the model state that achieved the best validation
        performance during training, ensuring we use the best version for evaluation.
        """
        model_path = os.path.join(self.results_dir, 'best_model.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print("No saved model found, using current model state")
    
    def debug_model_output(self, sample_text):
        """
        Debug function to analyze what the model is learning
        
        This method provides insights into the model's internal representations
        and output distributions, useful for debugging training issues.
        
        Args:
            sample_text (str): Sample text to analyze
        """
        print("="*50)
        print("MODEL DEBUGGING")
        print("="*50)
        
        # Tokenize sample input
        inputs = self.tokenizer(sample_text[:200], return_tensors='pt', max_length=512, truncation=True, padding=True).to(device)
        
        self.model.eval()
        with torch.no_grad():
            # Analyze encoder output
            memory = self.model.encode(inputs['input_ids'], inputs['attention_mask'])
            print(f"Encoder output shape: {memory.shape}")
            print(f"Encoder output mean: {memory.mean():.4f}")
            print(f"Encoder output std: {memory.std():.4f}")
            
            # Analyze decoder output with dummy target
            dummy_target = torch.zeros(1, 10).long().to(device)
            decoder_out = self.model.decode(dummy_target, memory)
            print(f"Decoder output shape: {decoder_out.shape}")
            print(f"Decoder logits mean: {decoder_out.mean():.4f}")
            print(f"Decoder logits std: {decoder_out.std():.4f}")
            
            # Analyze vocabulary distribution
            probs = F.softmax(decoder_out[0, -1, :], dim=-1)
            top_tokens = torch.topk(probs, 10)
            print(f"Top 10 token probabilities:")
            for i, (prob, token_id) in enumerate(zip(top_tokens[0], top_tokens[1])):
                token = self.tokenizer.decode([token_id.item()])
                print(f"  {i+1}. {token} ({prob:.4f})")
        
        print("="*50)
    
    def generate_summary(self, text, max_length=100):
        """
        Enhanced text generation with nucleus sampling and temperature control
        
        This method generates medical summaries using advanced sampling techniques
        for better quality and diversity. It implements nucleus (top-p) sampling
        and temperature scaling for controlled generation.
        
        Args:
            text (str): Input medical conversation to summarize
            max_length (int): Maximum length of generated summary
            
        Returns:
            str: Generated summary text
        """
        self.model.eval()               # Set model to evaluation mode
        
        # Tokenize and prepare input text
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            # Encode the input conversation
            memory = self.model.encode(inputs['input_ids'], inputs['attention_mask'])
            
            # Initialize generation with appropriate start token
            # Use pad token as start token (T5 convention)
            generated = [self.tokenizer.pad_token_id]
            
            # Autoregressive generation loop
            for step in range(max_length):
                # Prepare decoder input from generated tokens so far
                decoder_input = torch.tensor([generated]).to(device)
                
                # Safety check: ensure we don't exceed positional encoding limits
                if decoder_input.size(1) >= 2048:
                    break
                
                # Get next token logits from decoder
                output = self.model.decode(decoder_input, memory, memory_mask=inputs['attention_mask'])
                
                # Extract logits for next token prediction
                next_token_logits = output[0, -1, :]
                
                # Apply temperature scaling for controlled randomness
                # Lower temperature = more focused, higher = more diverse
                temperature = 0.7
                next_token_logits = next_token_logits / temperature
                
                # Convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Implement nucleus (top-p) sampling for better quality
                # This keeps only the most probable tokens that sum to top_p probability
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                top_p = 0.9
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                # Apply the filtering to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Prevent generating pad tokens in the middle of sequence
                if next_token == self.tokenizer.pad_token_id and len(generated) > 1:
                    break
                
                generated.append(next_token)
                
                # Stop if we hit the end-of-sequence token
                if next_token == self.tokenizer.eos_token_id:
                    break
            
            # Decode generated tokens to text, skip the initial pad token
            if len(generated) > 1:
                summary = self.tokenizer.decode(generated[1:], skip_special_tokens=True)
            else:
                summary = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            # Post-process the generated summary
            summary = summary.strip()
            
            # Simple deduplication to remove immediate repetitions
            words = summary.split()
            if len(words) > 3:
                cleaned_words = [words[0]]
                for i in range(1, len(words)):
                    if words[i] != words[i-1]:        # Remove consecutive duplicates
                        cleaned_words.append(words[i])
                summary = " ".join(cleaned_words)
            
            # Fallback if generation failed
            if not summary:
                summary = "No summary generated"
            
            return summary
    
    def extract_medical_entities(self, text):
        """
        Extract medical entities using spaCy and structure them in JSON format
        
        This method uses the SciSpaCy model to identify medical entities in text
        and organizes them into a structured format matching the dataset schema.
        This supports the goal of structured information extraction.
        
        Args:
            text (str): Input text to analyze for medical entities
            
        Returns:
            dict: Structured medical information following dataset JSON format
        """
        if self.nlp is None:
            return {}
        
        # Process text with spaCy NLP pipeline
        doc = self.nlp(text)
        
        # Initialize structured format matching the dataset's JSON schema
        # This format organizes medical information into clinically relevant categories
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
        
        # Initialize lists for entity collection
        symptoms_list = []
        treatments_list = []
        
        # Process each entity detected by spaCy
        for ent in doc.ents:
            entity_text = ent.text.strip()
            entity_label = ent.label_
            
            # Map spaCy entity types to medical information categories
            # Age detection using contextual clues
            if entity_label in ['CARDINAL'] and any(age_word in entity_text.lower() for age_word in ['year', 'old', 'age']):
                entities["patient information"]["age"] = entity_text
            
            # Gender detection using contextual clues
            elif entity_label in ['PERSON'] and any(gender in entity_text.lower() for gender in ['male', 'female', 'man', 'woman']):
                entities["patient information"]["sex"] = entity_text
            
            # Medication dosage detection
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
            
            # Medication/organization detection
            elif entity_label in ['ORG', 'PRODUCT']:
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
                # Default categorization as symptoms for uncategorized entities
                symptoms_list.append({
                    "name of symptom": entity_text,
                    "intensity of symptom": "None",
                    "location": "None",
                    "time": "None", 
                    "temporalisation": "None",
                    "behaviours affecting the symptom": "None",
                    "details": "None"
                })
        
        # Update entities structure with extracted information
        if symptoms_list:
            entities["symptoms"] = symptoms_list
        else:
            # Provide default structure if no symptoms found
            entities["symptoms"] = [{"name of symptom": "None", "intensity of symptom": "None", "location": "None", "time": "None", "temporalisation": "None", "behaviours affecting the symptom": "None", "details": "None"}]
        
        if treatments_list:
            entities["treatments"] = treatments_list
        else:
            # Provide default structure if no treatments found
            entities["treatments"] = [{"name": "None", "related condition": "None", "dosage": "None", "time": "None", "frequency": "None", "duration": "None", "reason for taking": "None", "reaction to treatment": "None", "details": "None"}]
        
        return entities
    
    def evaluate_on_test_set(self, test_data):
        """
        Comprehensive evaluation on test set using multiple metrics
        
        This method evaluates the trained model on unseen test data using
        standard metrics for text summarization: ROUGE and BERTScore.
        It provides both quantitative metrics and qualitative analysis.
        
        Args:
            test_data (list): Test dataset for evaluation
            
        Returns:
            dict: Comprehensive evaluation results
        """
        print("="*60)
        print("EVALUATING ENHANCED CUSTOM MEDICAL TRANSFORMER")
        print("="*60)
        
        # Load the best model checkpoint for evaluation
        self.load_best_model()
        
        # Generate predictions for all test samples
        predictions = []
        references = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            # Generate summary for each conversation
            predicted_summary = self.generate_summary(item['conversation'])
            predictions.append(predicted_summary)
            references.append(item['summary'])
        
        # Calculate comprehensive evaluation metrics
        results = self._calculate_metrics(predictions, references)
        results['test_samples'] = len(test_data)
        results['model_name'] = 'Enhanced Custom Medical Transformer'
        
        # Store results for later analysis
        self.evaluation_results = results
        
        # Save detailed results to files
        self._save_evaluation_results(test_data, predictions, references)
        
        return results
    
    def _calculate_metrics(self, predictions, references):
        """
        Calculate comprehensive evaluation metrics for text summarization
        
        This method computes multiple metrics to assess summary quality:
        - ROUGE scores for n-gram overlap
        - BERTScore for semantic similarity
        - Length statistics for analysis
        
        Args:
            predictions (list): Generated summaries
            references (list): Reference summaries
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        results = {}
        
        # ROUGE Scores - measure n-gram overlap between predictions and references
        # ROUGE-1: unigram overlap, ROUGE-2: bigram overlap, ROUGE-L: longest common subsequence
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        # Calculate ROUGE scores for each prediction-reference pair
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        # Compute mean and standard deviation for each ROUGE metric
        for metric in rouge_scores:
            results[f'{metric}_mean'] = np.mean(rouge_scores[metric])
            results[f'{metric}_std'] = np.std(rouge_scores[metric])
        
        # BERTScore - measures semantic similarity using BERT embeddings
        # This captures meaning beyond just n-gram overlap
        try:
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
            results['bertscore_precision'] = P.mean().item()
            results['bertscore_recall'] = R.mean().item()
            results['bertscore_f1'] = F1.mean().item()
        except Exception as e:
            print(f"BERTScore calculation failed: {e}")
            # Provide default values if BERTScore fails
            results['bertscore_precision'] = 0.0
            results['bertscore_recall'] = 0.0
            results['bertscore_f1'] = 0.0
        
        # Summary length statistics for analysis
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        results['avg_pred_length'] = np.mean(pred_lengths)
        results['avg_ref_length'] = np.mean(ref_lengths)
        results['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths)
        
        return results
    
    def _save_evaluation_results(self, test_data, predictions, references):
        """
        Save detailed evaluation results to files for further analysis
        
        This method saves both detailed results (with examples) and
        aggregated metrics for comprehensive analysis and reporting.
        
        Args:
            test_data (list): Original test data
            predictions (list): Generated predictions
            references (list): Reference summaries
        """
        # Create detailed results with examples
        detailed_results = []
        for i, item in enumerate(test_data):
            detailed_results.append({
                'conversation': item['conversation'],
                'reference_summary': references[i],
                'predicted_summary': predictions[i]
            })
        
        # Save detailed results as CSV for easy inspection
        df_results = pd.DataFrame(detailed_results)
        results_file = os.path.join(self.results_dir, 'detailed_results.csv')
        df_results.to_csv(results_file, index=False)
        
        # Save metrics as JSON for programmatic access
        metrics_file = os.path.join(self.results_dir, 'evaluation_metrics.json')
        combined_results = {
            'training_results': self.training_results,
            'evaluation_results': self.evaluation_results
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"Results saved to {self.results_dir}/")
    
    def print_results(self):
        """
        Print comprehensive results in a formatted, readable manner
        
        This method displays all key metrics and statistics from training
        and evaluation, providing a clear summary of model performance.
        """
        print("\n" + "="*60)
        print("ENHANCED CUSTOM MEDICAL TRANSFORMER RESULTS")
        print("="*60)
        
        if self.evaluation_results:
            print(f"\nMODEL INFORMATION:")
            print(f"  Model: {self.evaluation_results['model_name']}")
            print(f"  Type: Enhanced custom transformer (fine-tuned)")
            print(f"  Test Samples: {self.evaluation_results['test_samples']}")
            
            print(f"\n  ROUGE SCORES (n-gram overlap metrics):")
            print(f"    ROUGE-1: {self.evaluation_results.get('rouge1_mean', 0):.4f} (±{self.evaluation_results.get('rouge1_std', 0):.4f})")
            print(f"    ROUGE-2: {self.evaluation_results.get('rouge2_mean', 0):.4f} (±{self.evaluation_results.get('rouge2_std', 0):.4f})")
            print(f"    ROUGE-L: {self.evaluation_results.get('rougeL_mean', 0):.4f} (±{self.evaluation_results.get('rougeL_std', 0):.4f})")
            
            print(f"\n  BERTSCORE (semantic similarity metrics):")
            print(f"    Precision: {self.evaluation_results.get('bertscore_precision', 0):.4f}")
            print(f"    Recall: {self.evaluation_results.get('bertscore_recall', 0):.4f}")
            print(f"    F1: {self.evaluation_results.get('bertscore_f1', 0):.4f}")
            
            print(f"\n  SUMMARY STATISTICS:")
            print(f"    Avg Predicted Length: {self.evaluation_results['avg_pred_length']:.1f} words")
            print(f"    Avg Reference Length: {self.evaluation_results['avg_ref_length']:.1f} words")
            print(f"    Length Ratio: {self.evaluation_results['length_ratio']:.3f}")

def main():
    """
    Main execution function for the Enhanced Custom Medical Transformer
    
    This function orchestrates the complete pipeline:
    1. Data loading and preprocessing
    2. Model training with optimal hyperparameters
    3. Comprehensive evaluation on test set
    4. Results reporting and analysis
    
    The function uses optimized hyperparameters based on the architecture
    specifications and best practices for medical text summarization.
    
    Returns:
        CIM_Enhanced_Transformer: Trained transformer instance
    """
    print("Starting Enhanced Custom Medical Transformer Pipeline")
    print("=" * 60)
    
    # Initialize the transformer system
    transformer = CIM_Enhanced_Transformer()
    
    # Load and preprocess data
    # Using 15,000 samples for comprehensive training while maintaining feasibility
    print("Loading and preprocessing data...")
    data = transformer.load_and_preprocess_data(sample_size=15000)
    if not data:
        print("Failed to load data. Exiting.")
        return None
    
    # Split data into train/dev/test sets
    print("Splitting data...")
    train_data, dev_data, test_data = transformer.split_data(data)
    train_dataset, dev_dataset, test_dataset = transformer.create_datasets(train_data, dev_data, test_data)
    
    # Train model with enhanced settings and hyperparameters
    print("Training model...")
    transformer.train_model(
        train_dataset, dev_dataset,
        num_epochs=25,          # Extended training for better convergence
        batch_size=12,          # Balanced batch size for memory efficiency
        learning_rate=1e-4      # Optimized learning rate for transformer training
    )
    
    # Evaluate on test set with comprehensive metrics
    print("Evaluating model...")
    transformer.evaluate_on_test_set(test_data)
    
    # Display final results
    transformer.print_results()
    
    print("\nPipeline completed successfully!")
    return transformer

# Entry point for script execution
if __name__ == "__main__":
    # Execute the main pipeline
    transformer = main()