from sentence_transformers import SentenceTransformer, util
import numpy as np


def calculate_similarity_score(text1,text2):
    model = SentenceTransformer('stsb-roberta-large')

    # generate embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    # compute similarity scores of two embeddings
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_score

def preprocess_text(text):
    text = text.lower()
    return text

