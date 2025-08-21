import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import pandas as pd


class BaseSearcher:

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype('float32')

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        raise NotImplementedError


class FaissSearcher(BaseSearcher):
    """search using Faiss."""

    def __init__(self, embeddings: np.ndarray):
        super().__init__(embeddings)
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        normalized_embeddings = self.embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        self.index.add(normalized_embeddings)
        print(f"Faiss searcher initialized with {self.index.ntotal} vectors.")

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        query_vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector)
        similarities, indices = self.index.search(query_vector, top_k)
        return indices[0].tolist(), similarities[0].tolist()


class SklearnSearcher(BaseSearcher):
    """search using Scikit-learn (for comparison)."""

    def __init__(self, embeddings: np.ndarray):
        super().__init__(embeddings)
        print(f"Scikit-learn searcher initialized with {len(self.embeddings)} vectors.")

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        query_vector = query_vector.reshape(1, -1)
        sim_scores = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        return top_indices.tolist(), sim_scores[top_indices].tolist()


class SearchEngine:
    """main search engine class that orchestrates different ranking strategies."""

    def __init__(self, config):
        print("\n--- Step 2: Initializing Search Engine ---")
        self.config = config
        model_name = config['search']['embedding_model_in_use']
        sanitized_model_name = model_name.replace("/", "_")
        embedding_path = config['paths']['item_embeddings'].replace(".npy", f"_{sanitized_model_name}.npy")

        try:
            item_embeddings = np.load(embedding_path)
            # Initialize BM25 for keyword search
            items_df = pd.read_csv(config['paths']['processed_items'])
            corpus = items_df['combined_text'].fillna("").str.split().tolist()
            self.bm25 = BM25Okapi(corpus)
        except FileNotFoundError:
            print(f"Error: A required file was not found. Please run the data processing step first.")
            self.searcher = None
            return

        # Select the vector search backend
        backend = config['search']['backend']
        if backend == 'faiss':
            self.searcher = FaissSearcher(item_embeddings)
        elif backend == 'sklearn':
            self.searcher = SklearnSearcher(item_embeddings)
        else:
            raise ValueError(f"Unknown search backend: {backend}")

    def search(self, query_text: str, query_vector: np.ndarray) -> Tuple[List[int], List[float]]:
        """performs search based on the configured ranking strategy."""
        if self.searcher is None: return [], []
        top_k = self.config['settings']['top_k']
        strategy = self.config['search']['ranking_strategy']

        if strategy == 'cosine':
            return self.searcher.search(query_vector, top_k)

        query_vector = query_vector.reshape(1, -1)

        if strategy == 'euclidean':
            distances = euclidean_distances(query_vector, self.searcher.embeddings)[0]
            top_indices = np.argsort(distances)[:top_k]
            return top_indices.tolist(), distances[top_indices].tolist()

        elif strategy == 'advanced_hybrid':
            # 1. Calculate semantic scores (cosine similarity)
            cos_sim = cosine_similarity(query_vector, self.searcher.embeddings)[0]

            # 2. Calculate keyword scores using BM25
            tokenized_query = query_text.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            # Normalize BM25 scores to be in a comparable range (0-1)
            bm25_scores_normalized = bm25_scores / (bm25_scores.max() or 1.0)

            # 3. Fuse scores with configured weights
            weights = self.config['search']['hybrid_weights']
            hybrid_score = (cos_sim * weights['semantic_score']) + (bm25_scores_normalized * weights['keyword_score'])

            top_indices = np.argsort(hybrid_score)[::-1][:top_k]
            return top_indices.tolist(), hybrid_score[top_indices].tolist()

        else:
            raise ValueError(f"Unknown ranking strategy: {strategy}")