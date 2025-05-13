# doctor summary
# The purpose of this file is to recieve doctor vocal records, extract the relevant info, and present a simple summary for the 
# user to reference post treatment.
# 
# Pretrained model pipeline:
# T5 pre-trained transformer to perform abstractive summarization
# MedSpaCy to perform Named Entity Recognition

import torch
import medspacy

from transformers import T5Tokenizer, T5ForConditionalGeneration

