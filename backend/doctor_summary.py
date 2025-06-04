# doctor summary
# The purpose of this file is to receive doctor vocal records, extract the relevant info, and present a simple summary for the 
# user to reference post treatment.
# 
# Pretrained model pipeline:
# T5 pre-trained transformer to perform abstractive summarization
# MedSpaCy to perform Named Entity Recognition

import scispacy
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import whisper
import os
import re
import torch

class DoctorSummary:
    def __init__(self):
        self.nlp = None             # Scispacy model
        self.tokenizer = None       # T5 tokenizer
        self.model = None           # T5 model
        self.whisper_model = None   # Whisper model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_models()

    def initialize_models(self):
        """Initialize all required models"""
        print("Initializing models...")
        self.nlp = self.load_scispacy()
        self.tokenizer, self.model = self.load_t5()
        self.whisper_model = whisper.load_model("base")
        print("Models initialized successfully")

    def load_scispacy(self):
        """Load the MedSpaCy model with medical entity recognition"""
        nlp = spacy.load("en_core_sci_md")
        
        # Add entity ruler for better medical entity categorization
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        
        # Define comprehensive medical patterns
        patterns = [
            # Symptoms and Signs
            {"label": "SYMPTOM", "pattern": [
                {"LOWER": {"IN": [
                    "pain", "ache", "nausea", "fever", "headache", "fatigue", "dizziness",
                    "vomiting", "diarrhea", "constipation", "cough", "shortness", "breath",
                    "chest", "abdominal", "back", "joint", "muscle", "swelling", "rash",
                    "itching", "burning", "numbness", "tingling", "weakness", "tremor",
                    "seizure", "confusion", "drowsiness", "anxiety", "depression"
                ]}}
            ]},
            {"label": "SYMPTOM", "pattern": [
                {"LOWER": "chest"}, {"LOWER": "pain"}
            ]},
            {"label": "SYMPTOM", "pattern": [
                {"LOWER": "abdominal"}, {"LOWER": "pain"}
            ]},
            {"label": "SYMPTOM", "pattern": [
                {"LOWER": "short"}, {"LOWER": "of"}, {"LOWER": "breath"}
            ]},
            
            # Medications and Dosages
            {"label": "MEDICATION", "pattern": [
                {"LOWER": {"IN": [
                    "ibuprofen", "acetaminophen", "aspirin", "morphine", "codeine",
                    "prednisone", "antibiotics", "penicillin", "amoxicillin", "azithromycin",
                    "metformin", "insulin", "warfarin", "heparin", "lisinopril",
                    "atorvastatin", "metoprolol", "amlodipine", "omeprazole", "pantoprazole"
                ]}}
            ]},
            {"label": "MEDICATION", "pattern": [
                {"TEXT": {"REGEX": r".*mg$"}}
            ]},
            {"label": "MEDICATION", "pattern": [
                {"TEXT": {"REGEX": r".*ml$"}}
            ]},
            {"label": "MEDICATION", "pattern": [
                {"TEXT": {"REGEX": r".*mcg$"}}
            ]},
            {"label": "MEDICATION", "pattern": [
                {"TEXT": {"REGEX": r".*units?$"}}
            ]},
            
            # Procedures and Tests
            {"label": "PROCEDURE", "pattern": [
                {"LOWER": {"IN": [
                    "surgery", "operation", "examination", "x-ray", "ct", "mri", "ultrasound",
                    "biopsy", "endoscopy", "colonoscopy", "catheterization", "dialysis",
                    "transfusion", "vaccination", "injection", "infusion", "dressing",
                    "suturing", "casting", "splinting", "reduction", "drainage"
                ]}}
            ]},
            {"label": "PROCEDURE", "pattern": [
                {"LOWER": "blood"}, {"LOWER": "test"}
            ]},
            {"label": "PROCEDURE", "pattern": [
                {"LOWER": "follow"}, {"LOWER": "up"}
            ]},
            
            # Diagnoses and Conditions
            {"label": "DIAGNOSIS", "pattern": [
                {"LOWER": {"IN": [
                    "appendicitis", "pneumonia", "diabetes", "hypertension", "infection",
                    "fracture", "arthritis", "asthma", "cancer", "stroke", "heart attack",
                    "angina", "anemia", "thyroid", "kidney", "liver", "failure",
                    "depression", "anxiety", "migraine", "seizure", "allergy"
                ]}}
            ]},
            
            # Instructions and Recommendations
            {"label": "INSTRUCTION", "pattern": [
                {"LOWER": "take"}, {"IS_ALPHA": True}
            ]},
            {"label": "INSTRUCTION", "pattern": [
                {"LOWER": "avoid"}
            ]},
            {"label": "INSTRUCTION", "pattern": [
                {"LOWER": "return"}, {"LOWER": "if"}
            ]},
            {"label": "INSTRUCTION", "pattern": [
                {"LOWER": "follow"}, {"LOWER": "up"}
            ]},
            {"label": "INSTRUCTION", "pattern": [
                {"LOWER": "continue"}
            ]},
            {"label": "INSTRUCTION", "pattern": [
                {"LOWER": "stop"}
            ]},
            
            # Vital Signs and Measurements
            {"label": "VITAL", "pattern": [
                {"LOWER": {"IN": [
                    "blood pressure", "heart rate", "pulse", "temperature", "temp",
                    "respiratory rate", "oxygen", "saturation", "weight", "height",
                    "bmi", "glucose", "sugar"
                ]}}
            ]},
            {"label": "VITAL", "pattern": [
                {"TEXT": {"REGEX": r"^\d{2,3}/\d{2,3}$"}}  # Blood pressure pattern
            ]},
            {"label": "VITAL", "pattern": [
                {"TEXT": {"REGEX": r"^\d{2,3}\.?\d?$"}}  # Temperature pattern
            ]}
        ]
        
        ruler.add_patterns(patterns)
        return nlp

    def load_t5(self):
        """Load the T5 model and tokenizer"""
        # Use T5-base for better balance of quality and speed
        model_name = "t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(self.device)
        return tokenizer, model

    def preprocess_medical_text(self, text):
        """Clean and preprocess the input text"""
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Fix sentence boundaries
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Handle common medical abbreviations and ensure proper spacing
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms)\.', r'\1', text)  # Remove periods from titles
        text = re.sub(r'(\d+)\s*(mg|ml|cc|mcg|units?)', r'\1\2', text)  # Normalize dosages
        
        return text

    def extract_medical_context_from_entities(self, entities, text):
        """Convert extracted entities into categorized medical context"""
        context = {
            'symptoms': [],
            'medications': [],
            'procedures': [],
            'diagnoses': [],
            'instructions': [],
            'all_entities': entities
        }
        
        # Categorize entities based on their assigned categories
        for entity in entities:
            category = entity['category']
            text_content = entity['text']
            
            if category == 'SYMPTOM':
                context['symptoms'].append(text_content)
            elif category == 'MEDICATION':
                context['medications'].append(text_content)
            elif category == 'PROCEDURE':
                context['procedures'].append(text_content)
            elif category == 'DIAGNOSIS':
                context['diagnoses'].append(text_content)
            elif category == 'INSTRUCTION':
                context['instructions'].append(text_content)
        
        # Pattern-based instruction extraction (as fallback for missed instructions)
        instruction_patterns = [
            r'take\s+[\w\s]+?(?:daily|twice|once|every|as needed)',
            r'follow\s+up\s+(?:with|in)[\w\s]+',
            r'return\s+if[\w\s]+',
            r'avoid\s+[\w\s]+',
            r'continue\s+[\w\s]+',
            r'stop\s+[\w\s]+?if'
        ]
        
        for pattern in instruction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            context['instructions'].extend(matches)
        
        # Remove duplicates while preserving order
        for key in context:
            if key != 'all_entities':
                context[key] = list(dict.fromkeys(context[key]))  # Remove duplicates
        
        return context

    def create_structured_prompt(self, text, context):
        """Create a structured prompt that encourages abstractive summarization using extracted entities"""
        
        # Build context-aware prompt with actual entity information
        prompt_parts = [
            "Medical Summary:",
            "Patient presents with " + (context['symptoms'][0] if context['symptoms'] else "symptoms") + ".",
            "Diagnosis: " + (context['diagnoses'][0] if context['diagnoses'] else "pending") + ".",
            "Treatment includes " + (context['procedures'][0] if context['procedures'] else "ongoing care") + ".",
            "Medications: " + (", ".join(context['medications'][:2]) if context['medications'] else "none prescribed") + ".",
            "Follow-up: " + (context['instructions'][0] if context['instructions'] else "routine follow-up") + ".",
            "",
            "Original consultation:",
            text,
            "",
            "Generate a complete medical summary:"
        ]
        
        return "\n".join(prompt_parts)

    def transcribe_audio(self, audio_path):
        """Transcribe audio file using Whisper"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        result = self.whisper_model.transcribe(audio_path, fp16=False)
        return result["text"]

    def extract_entities(self, text):
        """Extract named entities from text using SciSpaCy with better categorization"""
        doc = self.nlp(text)
        
        # Return entities with additional categorization info
        entities = []
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'category': self.categorize_entity(ent.text, ent.label_)
            }
            entities.append(entity_info)
        
        return entities
    
    def categorize_entity(self, entity_text, entity_label):
        """Enhanced entity categorization with context analysis"""
        text_lower = entity_text.lower()
        
        # If we already have a specific label, use it
        if entity_label in ['SYMPTOM', 'MEDICATION', 'PROCEDURE', 'DIAGNOSIS', 'INSTRUCTION', 'VITAL']:
            return entity_label
        
        # Enhanced keyword-based categorization
        symptom_keywords = {
            'pain', 'ache', 'nausea', 'fever', 'headache', 'fatigue', 'dizziness',
            'vomiting', 'diarrhea', 'constipation', 'cough', 'shortness', 'breath',
            'chest', 'abdominal', 'back', 'joint', 'muscle', 'swelling', 'rash'
        }
        
        medication_keywords = {
            'mg', 'ml', 'mcg', 'unit', 'tablet', 'capsule', 'dose', 'prescription',
            'ibuprofen', 'acetaminophen', 'aspirin', 'morphine', 'codeine', 'prednisone'
        }
        
        procedure_keywords = {
            'surgery', 'operation', 'examination', 'test', 'scan', 'x-ray', 'ct', 'mri',
            'ultrasound', 'biopsy', 'endoscopy', 'colonoscopy', 'catheterization'
        }
        
        diagnosis_keywords = {
            'appendicitis', 'pneumonia', 'diabetes', 'hypertension', 'infection',
            'fracture', 'arthritis', 'asthma', 'cancer', 'stroke', 'heart attack'
        }
        
        vital_keywords = {
            'blood pressure', 'heart rate', 'pulse', 'temperature', 'temp',
            'respiratory rate', 'oxygen', 'saturation', 'weight', 'height', 'bmi'
        }
        
        # Check for keywords in the entity text
        if any(keyword in text_lower for keyword in symptom_keywords):
            return 'SYMPTOM'
        elif any(keyword in text_lower for keyword in medication_keywords):
            return 'MEDICATION'
        elif any(keyword in text_lower for keyword in procedure_keywords):
            return 'PROCEDURE'
        elif any(keyword in text_lower for keyword in diagnosis_keywords):
            return 'DIAGNOSIS'
        elif any(keyword in text_lower for keyword in vital_keywords):
            return 'VITAL'
        
        # Check for numeric patterns that might indicate measurements
        if re.search(r'\d+\s*(mg|ml|mcg|units?|kg|lb|cm|in)', text_lower):
            return 'MEDICATION'
        if re.search(r'\d{2,3}/\d{2,3}', text_lower):  # Blood pressure pattern
            return 'VITAL'
        if re.search(r'\d{2,3}\.?\d?', text_lower):  # Temperature pattern
            return 'VITAL'
        
        return 'MEDICAL_TERM'  # Default category for other medical terms

    def summarize(self, text, entities=None):
        """Generate an abstractive summary of the text using extracted entities"""
        print('Performing abstractive summarization...')
        
        # Preprocess the text
        text = self.preprocess_medical_text(text)
        
        # If entities aren't provided, extract them
        if entities is None:
            entities = self.extract_entities(text)
        
        # Convert entities to medical context
        context = self.extract_medical_context_from_entities(entities, text)
        
        # Create structured prompt using the entity context
        input_text = self.create_structured_prompt(text, context)
        
        # Tokenize with appropriate length limits
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate summary with parameters optimized for abstractive summarization
        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids,
                max_length=150,      # Shorter to prevent repetition
                min_length=50,       # Ensure sufficient detail
                num_beams=4,         # Balanced beam search
                length_penalty=1.0,  # Neutral length penalty
                early_stopping=True,
                no_repeat_ngram_size=3,  # Prevent repetition
                temperature=0.6,     # Lower temperature for more focused generation
                top_k=30,           # More focused token selection
                top_p=0.8,          # More focused nucleus sampling
                do_sample=True,     # Enable sampling
                repetition_penalty=1.5,  # Strong repetition penalty
                num_return_sequences=1
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Post-process the summary to ensure abstractive nature
        summary = self.post_process_summary(summary, text)
        
        print("Generated summary:", summary)
        return summary

    def post_process_summary(self, summary, original_text):
        """Post-process the summary to ensure quality and abstractive nature"""
        
        # Clean up the summary
        summary = summary.strip()
        summary = re.sub(r'\s+', ' ', summary)
        
        # Remove any prompt-like text or instructions
        summary = re.sub(r'^(medical summary|summary|generate|write|create):\s*', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'^(the\s+)?(patient|doctor|medical)\s+(consultation|record|note)?\s*:?\s*', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'^(focus on|key points|main points):\s*', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'^(you are|this is|here is|the following|as follows):\s*', '', summary, flags=re.IGNORECASE)
        
        # Remove any bullet points or list markers
        summary = re.sub(r'^[-•*]\s*', '', summary, flags=re.MULTILINE)
        
        # Remove any remaining prompt artifacts
        summary = re.sub(r'^(include|focus|write|generate|create|summarize).*?[:.]', '', summary, flags=re.IGNORECASE)
        
        # Ensure the summary starts with a proper sentence
        if not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure proper sentence endings
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        # Remove any remaining template-like text
        summary = re.sub(r'^(patient presents with|diagnosis:|treatment includes|medications:|follow-up:)\s*', '', summary, flags=re.IGNORECASE)
        
        # Check for extractive patterns and filter them out
        original_sentences = re.split(r'[.!?]+', original_text.lower())
        summary_sentences = re.split(r'[.!?]+', summary.lower())
        
        filtered_sentences = []
        for summary_sent in summary_sentences:
            summary_sent = summary_sent.strip()
            if len(summary_sent) < 10:  # Skip very short sentences
                continue
                
            # Check if this sentence is too similar to any original sentence
            is_extractive = False
            for orig_sent in original_sentences:
                orig_sent = orig_sent.strip()
                if len(orig_sent) < 10:
                    continue
                    
                # Calculate similarity (simple word overlap)
                summary_words = set(summary_sent.split())
                orig_words = set(orig_sent.split())
                
                if len(summary_words) > 0:
                    overlap = len(summary_words.intersection(orig_words))
                    similarity = overlap / len(summary_words)
                    
                    # If more than 70% of words match, consider it extractive
                    if similarity > 0.7:
                        is_extractive = True
                        break
            
            if not is_extractive:
                filtered_sentences.append(summary_sent.capitalize())
        
        # Reconstruct summary from non-extractive sentences
        if filtered_sentences:
            summary = '. '.join(filtered_sentences)
            if not summary.endswith('.'):
                summary += '.'
        # else:
            # If all sentences were filtered, use a fallback approach
            # summary = self.generate_fallback_summary(original_text)
        
        return summary

    def generate_fallback_summary(self, text):
        """Generate a basic abstractive summary when the main approach fails"""
        # Simple keyword-based approach as fallback
        context = self.extract_medical_context(text)
        
        parts = []
        if context['diagnoses']:
            parts.append(f"Patient presents with {', '.join(context['diagnoses'][:2])}")
        
        if context['procedures']:
            parts.append(f"Treatment includes {', '.join(context['procedures'][:2])}")
        
        if context['medications']:
            parts.append(f"Prescribed medications: {', '.join(context['medications'][:2])}")
        
        if context['instructions']:
            parts.append("Patient advised to follow specific care instructions")
        
        if not parts:
            parts.append("Medical consultation completed with patient care plan established")
        
        return '. '.join(parts) + '.'

    def process_audio(self, audio_path):
        """Process an audio file: transcribe, extract entities, and summarize"""
        try:
            # Transcribe the audio
            transcription = self.transcribe_audio(audio_path)
            print("Transcription:", transcription)
            
            # Extract entities with categorization (single source of truth)
            entities = self.extract_entities(transcription)
            print("Entities found:")
            for ent in entities:
                print(f"  - {ent['text']} (Label: {ent['label']}, Category: {ent['category']})")
            
            # Generate summary using the extracted entities
            summary = self.summarize(transcription, entities)
            
            return {
                'transcription': transcription,
                'entities': entities,
                'summary': summary
            }
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise

    def process_text(self, text):
        """Process text input: extract entities and generate summary"""
        try:
            # Extract entities with categorization (single source of truth)
            entities = self.extract_entities(text)
            print("Entities found:")
            for ent in entities:
                print(f"  - {ent['text']} (Label: {ent['label']}, Category: {ent['category']})")
            
            # Generate summary using the extracted entities
            summary = self.summarize(text, entities)
            
            return {
                'summary': summary,
                'entities': entities,
                'original_text': text
            }
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            raise

def main():
    # Example usage with sample medical text
    doctor_summary = DoctorSummary()
    
    # Test with sample medical consultation text
    sample_text = """
    The patient presents with acute abdominal pain in the lower right quadrant that began approximately 6 hours ago. 
    Physical examination reveals tenderness and guarding in the right iliac fossa. Temperature is elevated at 101.2°F. 
    White blood cell count is 15,000. Based on clinical presentation and laboratory findings, the diagnosis is acute appendicitis. 
    The patient has been scheduled for laparoscopic appendectomy tonight. Pre-operative antibiotics have been administered. 
    Post-operative instructions include bed rest for 24 hours, clear liquids initially, and gradual return to normal diet. 
    Follow-up appointment in one week. Patient should return immediately if experiencing severe pain, fever, or signs of infection.
    """
    
    result = doctor_summary.process_text(sample_text)
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print("\nOriginal Text Length:", len(sample_text.split()))
    print("\nSummary:", result['summary'])
    print("\nSummary Length:", len(result['summary'].split()))
    print("\nExtracted Entities:")
    for entity in result['entities']:
        print(f"  - {entity['text']} (Label: {entity['label']}, Category: {entity['category']})")

if __name__ == "__main__":
    main()