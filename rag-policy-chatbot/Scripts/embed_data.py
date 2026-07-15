from typing import List

import numpy as np

from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def generate_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    clean_texts = [text if isinstance(text, str) else "" for text in texts]
    embeddings = model.encode(clean_texts, show_progress_bar=False, convert_to_numpy=True)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    return embeddings
