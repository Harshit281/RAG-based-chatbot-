import os
import pickle
from typing import Tuple

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self.index = self._build_index(self.embeddings)

    def _build_index(self, embeddings: np.ndarray):
        if _HAS_FAISS:
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            return index
        from sklearn.neighbors import NearestNeighbors
        self._sklearn_index = NearestNeighbors(n_neighbors=min(10, len(embeddings)), algorithm='auto', metric='euclidean')
        self._sklearn_index.fit(embeddings)
        return None

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        query_embedding = np.asarray(query_embedding).astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        if _HAS_FAISS:
            distances, indices = self.index.search(query_embedding, top_k)
            return distances, indices
        distances, indices = self._sklearn_index.kneighbors(query_embedding, n_neighbors=min(top_k, len(self.embeddings)))
        return distances, indices


def save_index(index_obj: VectorStore, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _HAS_FAISS:
        faiss.write_index(index_obj.index, path)
    else:
        with open(path, 'wb') as file:
            pickle.dump(index_obj, file)


def load_index(path: str, embeddings: np.ndarray) -> VectorStore:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    if _HAS_FAISS:
        index = faiss.read_index(path)
        store = VectorStore(embeddings)
        store.index = index
        return store
    with open(path, 'rb') as file:
        return pickle.load(file)
