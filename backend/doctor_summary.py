# doctor summary
# The purpose of this file is to recieve doctor vocal records, extract the relevant info, and present a simple summary for the 
# user to reference post treatment.
# 
# Pretrained model pipeline:
# T5 pre-trained transformer to perform abstractive summarization
# MedSpaCy to perform Named Entity Recognition

import scispacy
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration
import whisper
import os

def transcribe_audio(audio_path):
    # Load the Whisper model (you can choose different sizes: tiny, base, small, medium, large)
    model = whisper.load_model("base")
    
    # Transcribe the audio file with fp16=False to suppress the warning
    result = model.transcribe(audio_path, fp16=False)
    
    return result["text"]

# Use the full path to the audio file



# Load the MedSpaCy model
def load_scispacy():
    nlp = spacy.load("en_core_sci_md")
    return nlp

def NER(transcription, nlp):
    doc = nlp(transcription)
    print(doc.ents)
    return doc.ents

def load_t5():
    tokenizer = T5Tokenizer.from_pretrained("t5-large", legacy=True)
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    return tokenizer, model

def summarize(text, tokenizer, model):
    # pass NER outputs into summarizer along with raw text
    print('summarizing')

    # Prepend task prefix
    input_text = "summarize: " + text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=500, num_beams=5, length_penalty=1.2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(summary)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(current_dir, "test.mp3")
    transcription = transcribe_audio(audio_file)
    
    nlp = load_scispacy()
    tokenizer, model = load_t5()
    NER(transcription, nlp)
    summarize(transcription, tokenizer, model)

if __name__ == "__main__":
    main()