# chatbot_intake
# The purpose of this file is to recieve patient info and feed into the chatbot for 
# text summarization. 
# 
# Pretrained model pipeline:
# T5 pre-trained transformer to perform abstractive summarization
# MedSpaCy to perform Named Entity Recognition
import scispacy
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration


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
    print(doc.ents)

    
def summarize():
    # pass NER outputs into summarizer along with raw text
    print('summarizing')



    tokenizer = T5Tokenizer.from_pretrained("t5-large", legacy=True)
    model = T5ForConditionalGeneration.from_pretrained("t5-large")

    # Your input text
    text = """

    """

    # Prepend task prefix
    input_text = "summarize: " + text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=5000, num_beams=5, length_penalty=0.7, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(summary)


def main():
    NER()
    summarize()

if __name__ == '__main__':
    main()
