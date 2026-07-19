from typing import List, Dict, Optional

import numpy as np

from Scripts.embed_data import generate_embeddings
from Scripts.vector_store import VectorStore


def _build_query_variants(query: str) -> List[str]:
    variants = [query.strip()]
    if query:
        variants += [
            f"{query.strip()} policy",
            f"company policy for {query.strip()}",
            f"guidelines for {query.strip()}"
        ]
    return list(dict.fromkeys([variant for variant in variants if variant]))


def retrieve(
    query: str,
    embedding_model,
    vector_store: VectorStore,
    chunks: List[Dict],
    top_k: int = 5,
    multi_query: bool = True,
) -> List[Dict]:
    if not query:
        return []

    query_variants = _build_query_variants(query) if multi_query else [query.strip()]
    all_matches = {}
    for variant in query_variants:
        embeddings = generate_embeddings(embedding_model, [variant])
        distances, indices = vector_store.search(embeddings, top_k=top_k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(chunks):
                continue
            item = all_matches.get(idx, {'count': 0, 'distance_sum': 0.0, 'chunk': chunks[idx]})
            item['count'] += 1
            item['distance_sum'] += float(dist)
            all_matches[idx] = item

    ranked = sorted(
        all_matches.values(),
        key=lambda item: (item['distance_sum'] / item['count'], -item['count'])
    )
    return [item['chunk'] for item in ranked[:top_k]]
