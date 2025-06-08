from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from tqdm import tqdm

class Retriever(ABC):
    """Abstract base class for document retrieval."""
    
    @abstractmethod
    def index_documents(self, document_chunks: List[Dict]):
        """Index the document chunks for retrieval."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the top-k relevant document chunks for a query."""
        pass

class TFIDFRetriever(Retriever):
    def __init__(self):
        """Initialize the TF-IDF retriever."""
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_chunks = []
        self.tfidf_matrix = None
        
    def index_documents(self, document_chunks: List[Dict]):
        """
        Index the document chunks using TF-IDF.
        
        Args:
            document_chunks: List of document chunks with text and metadata
        """
        self.document_chunks = document_chunks
        texts = [chunk["text"] for chunk in document_chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k relevant document chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of most relevant document chunks with scores
        """
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices of top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                "text": self.document_chunks[idx]["text"],
                "metadata": self.document_chunks[idx]["metadata"],
                "score": float(similarities[idx])
            })
        
        return results

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, document_chunks: List[Dict]):
        """Add vectors and their associated documents to the store."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for the top-k most similar vectors."""
        pass

class FAISSVectorStore(VectorStore):
    """Vector store implementation using FAISS."""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """Initialize the FAISS vector store.
        
        Args:
            dimension: Dimension of the vectors to store
            index_type: Type of FAISS index to use ('flat', 'ivf', 'hnsw')
        """
        import faiss
        
        # Select the appropriate index type
        if index_type == "flat":
            # Simple but exact flat index (L2 distance)
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # IVF (Inverted File Index) for faster but approximate search
            # Using 4*sqrt(n) centroids as a rule of thumb, but will use 100 to start
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_L2)
            # IVF requires training before adding vectors
            self.needs_training = True
        elif index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) for efficient search
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.document_chunks = []
        self.needs_training = hasattr(self, 'needs_training') and self.needs_training
        self.index_type = index_type
        self.dimension = dimension
        
    def train(self, vectors: np.ndarray):
        """Train the index if required (for IVF indexes)."""
        if self.needs_training and not self.index.is_trained:
            vectors = vectors.astype(np.float32)
            self.index.train(vectors)
        
    def add_vectors(self, vectors: np.ndarray, document_chunks: List[Dict]):
        """Add vectors and their associated documents to the store."""
        # Make sure vectors are in float32 format
        vectors = vectors.astype(np.float32)
        
        # Train the index if needed
        if self.needs_training:
            self.train(vectors)
        
        # Add vectors to the index
        self.index.add(vectors)
        self.document_chunks = document_chunks
        
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for the top-k most similar vectors."""
        # Make sure query vector is in float32 format and reshaped for FAISS
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        # Perform the search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.document_chunks):  # Ensure index is valid
                results.append({
                    "text": self.document_chunks[idx]["text"],
                    "metadata": self.document_chunks[idx]["metadata"],
                    "score": float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                })
        
        return results

class DenseRetriever(Retriever):
    """Retriever that uses dense embeddings and a vector store."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 vector_store: Optional[VectorStore] = None,
                 vector_store_config: Optional[Dict[str, Any]] = None,
                 batch_size: int = 32):
        """Initialize the dense retriever with embedding model and vector store.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            vector_store: Vector store to use (if None, will create a FAISS store)
            vector_store_config: Configuration options for the FAISS vector store
            batch_size: Batch size for encoding documents
        """
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        self.document_chunks = []
        self.model_name = model_name
        self.batch_size = batch_size
        
        # If no vector store is provided, create a FAISS vector store
        if vector_store is None:
            dimension = self.model.get_sentence_embedding_dimension()
            config = vector_store_config or {}
            self.vector_store = FAISSVectorStore(dimension, **config)
        else:
            self.vector_store = vector_store
        
    def index_documents(self, document_chunks: List[Dict]):
        """Index the document chunks using embeddings and the vector store."""
        self.document_chunks = document_chunks
        texts = [chunk["text"] for chunk in document_chunks]
        
        # Generate embeddings for all document chunks using batches for efficiency
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batched embeddings
        if len(all_embeddings) > 1:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = all_embeddings[0]
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, document_chunks)
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the top-k relevant document chunks for a query."""
        # Generate embedding for the query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Search the vector store
        return self.vector_store.search(query_embedding, top_k)

class HybridRetriever(Retriever):
    """A retriever that combines results from multiple retrievers."""
    
    def __init__(self, retrievers: Dict[str, Retriever], weights: Optional[Dict[str, float]] = None):
        """Initialize with multiple retrievers.
        
        Args:
            retrievers: Dictionary mapping retriever names to retriever instances
            weights: Dictionary mapping retriever names to their weights (default: equal weights)
        """
        self.retrievers = retrievers
        
        # Set weights (default: equal weights for all retrievers)
        if weights is None:
            self.weights = {name: 1.0 / len(retrievers) for name in retrievers}
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {name: weight / total for name, weight in weights.items()}
            
            # Ensure all retrievers have weights
            for name in retrievers:
                if name not in self.weights:
                    self.weights[name] = 0.0
    
    def index_documents(self, document_chunks: List[Dict]):
        """Index documents in all retrievers."""
        for retriever in self.retrievers.values():
            retriever.index_documents(document_chunks)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve documents using all retrievers and merge results."""
        # Get results from each retriever
        all_results = {}
        for name, retriever in self.retrievers.items():
            results = retriever.retrieve(query, top_k=top_k * 2)  # Get more results to have enough for merging
            
            # Add to all_results, keyed by document ID (using text as ID for simplicity)
            for result in results:
                doc_id = result.get("metadata", {}).get("source", "") + "::" + result["text"][:100]
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "text": result["text"],
                        "metadata": result["metadata"],
                        "scores": {}
                    }
                
                # Store score from this retriever
                all_results[doc_id]["scores"][name] = result["score"]
        
        # Calculate weighted scores
        for doc_id, result in all_results.items():
            weighted_score = 0.0
            for name, score in result["scores"].items():
                weighted_score += score * self.weights.get(name, 0.0)
            
            result["score"] = weighted_score
        
        # Sort by weighted score and take top-k
        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )[:top_k]
        
        # Remove the individual scores
        for result in sorted_results:
            result.pop("scores", None)
        
        return sorted_results

class ReRankingRetriever(Retriever):
    """A retriever that re-ranks the results from a base retriever using a cross-encoder model."""
    
    def __init__(self, 
                 base_retriever: Retriever,
                 cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 initial_top_k: int = 10):
        """Initialize with a base retriever and a cross-encoder model for re-ranking.
        
        Args:
            base_retriever: The underlying retriever to get initial results from
            cross_encoder_name: The name of the cross-encoder model to use for re-ranking
            initial_top_k: How many documents to retrieve initially before re-ranking
        """
        self.base_retriever = base_retriever
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        self.initial_top_k = initial_top_k
        
    def index_documents(self, document_chunks: List[Dict]):
        """Pass indexing to the base retriever."""
        self.base_retriever.index_documents(document_chunks)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve and re-rank documents for the query.
        
        Args:
            query: The search query
            top_k: Number of top results to return after re-ranking
            
        Returns:
            List of re-ranked document chunks with scores
        """
        # Get initial results from base retriever (retrieve more than we need)
        initial_k = max(self.initial_top_k, top_k * 2)
        initial_results = self.base_retriever.retrieve(query, top_k=initial_k)
        
        if not initial_results:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, doc["text"]) for doc in initial_results]
        
        # Score the pairs with cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Sort results by cross-encoder scores
        for i, score in enumerate(scores):
            initial_results[i]["score"] = float(score)
        
        reranked_results = sorted(initial_results, key=lambda x: x["score"], reverse=True)
        
        # Return top_k results
        return reranked_results[:top_k]

def get_retriever(retriever_type: str = "tfidf", **kwargs) -> Retriever:
    """Factory function to create a retriever.
    
    Args:
        retriever_type: Type of retriever to create ('tfidf', 'dense', 'hybrid', 'rerank')
        **kwargs: Additional arguments to pass to the retriever constructor
    
    Returns:
        An instance of a Retriever subclass
    """
    if retriever_type.lower() == "tfidf":
        return TFIDFRetriever()
    elif retriever_type.lower() == "dense":
        vector_store_config = kwargs.pop('vector_store_config', {})
        return DenseRetriever(
            vector_store_config=vector_store_config,
            **kwargs
        )
    elif retriever_type.lower() == "hybrid":
        # Hybrid requires specification of retrievers to combine
        retrievers = kwargs.get('retrievers', {
            'tfidf': TFIDFRetriever(),
            'dense': DenseRetriever()
        })
        weights = kwargs.get('weights', None)
        return HybridRetriever(retrievers, weights)
    elif retriever_type.lower() == "rerank":
        # Get the base retriever type
        base_retriever_type = kwargs.pop('base_retriever_type', 'dense')
        # Additional args for base retriever
        base_retriever_kwargs = kwargs.pop('base_retriever_kwargs', {})
        # Create base retriever
        base_retriever = get_retriever(base_retriever_type, **base_retriever_kwargs)
        # Create re-ranking retriever
        return ReRankingRetriever(base_retriever, **kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
