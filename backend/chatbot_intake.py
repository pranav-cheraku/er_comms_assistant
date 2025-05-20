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
        New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
        A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
        Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
        In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
        Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
        2010 marriage license application, according to court documents.
        Prosecutors said the marriages were part of an immigration scam.
        On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
        After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
        Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
        All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
        Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
        Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
        The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
        Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
        Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
        If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """

    # Prepend task prefix
    input_text = "summarize the following text: " + text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=150, num_beams=5, length_penalty=1.2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(summary)


def main():
    NER()
    summarize()

if __name__ == '__main__':
    main()