from transformers import AutoTokenizer,pipeline
import spacy
from sentence_transformers import SentenceTransformer
import re

# Lang id + normalization
lang_id = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
#
def normalize_query(text: str) -> str:
    text=text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

embedder = SentenceTransformer('intfloat/multilingual-e5-large')