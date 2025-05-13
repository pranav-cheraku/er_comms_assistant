# chatbot_intake
# The purpose of this file is to recieve patient info and feed into the chatbot for 
# text summarization. 
# 
# Pretrained model pipeline:
# T5 pre-trained transformer to perform abstractive summarization
# MedSpaCy to perform Named Entity Recognition

import torch
import medspacy

from transformers import T5Tokenizer, T5ForConditionalGeneration

def NER():
    nlp = medspacy.load()
    sample_text = "The patient denies chest pain but has a history of hypertension."
    # need a way to compile all user text into one conversation string
    doc = nlp(sample_text)

    for ent in doc.ents:
        print(ent.text, ent.label_)

def summarize():
    ner = NER()
    
    # pass NER outputs into summarizer along with raw text
    print('summarizing')

def main():
    NER()

if __name__ == '__main__':
    main()