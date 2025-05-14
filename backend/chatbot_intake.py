# chatbot_intake
# The purpose of this file is to recieve patient info and feed into the chatbot for 
# text summarization. 
# 
# Pretrained model pipeline:
# T5 pre-trained transformer to perform abstractive summarization
# MedSpaCy to perform Named Entity Recognition
import scispacy
import spacy

def NER():
    nlp = spacy.load("en_core_sci_md")
    text = """
    Myeloid derived suppressor cells (MDSC) are immature 
    myeloid cells with immunosuppressive activity. 
    They accumulate in tumor-bearing mice and humans 
    with different types of cancer, including hepatocellular 
    carcinoma (HCC).
    """
    doc = nlp(text)

    print(list(doc.sents))

def summarize():
    ner = NER()
    
    # pass NER outputs into summarizer along with raw text
    print('summarizing')

def main():
    NER()

if __name__ == '__main__':
    main()