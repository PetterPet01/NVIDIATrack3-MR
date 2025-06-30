# Imports
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer
import threading
from contextlib import contextmanager

# --- The RWLock Helper Class (unchanged) ---
class RWLock:
    """
    A simple Read-Write Lock implementation. Allows for multiple concurrent readers
    or a single exclusive writer.
    """
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        """Acquire a read lock. Blocks only if a write lock is held."""
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """Release a read lock."""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """Acquire a write lock. Blocks until there are no readers."""
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """Release a write lock."""
        self._read_ready.release()


# --- The Corrected MilvusManager Class ---
class MilvusManager:
    """
    A thread-safe, specialized Milvus manager for performing few-shot similarity 
    searches on a pre-built text-based instruction database.

    This class uses a Singleton pattern to ensure that the expensive model and 
    database connection are initialized only once and shared across all threads.
    It employs a Read-Write Lock to allow for high-performance concurrent searches 
    while ensuring that database build operations are exclusive and atomic.
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @contextmanager
    def _read_lock(self):
        """Context manager for acquiring a read lock."""
        self._rw_lock.acquire_read()
        try:
            yield
        finally:
            # CORRECTED: The release method must be called on the _rw_lock object.
            self._rw_lock.release_read()

    @contextmanager
    def _write_lock(self):
        """Context manager for acquiring a write lock."""
        self._rw_lock.acquire_write()
        try:
            yield
        finally:
            # CORRECTED: The release method must be called on the _rw_lock object.
            self._rw_lock.release_write()

    def __init__(self, host="localhost", port="19530"):
        with self._lock:
            if self._initialized:
                return
            
            self.model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
            self.collection_name = "fewshot_retrieve"
            self.embedding_dim = 384
            self.metric_type = "COSINE"
            self.index_type = "HNSW"
            self._rw_lock = RWLock()
            
            print(f"Connecting to Milvus at {host}:{port}...")
            connections.connect(alias="default", host=host, port=port)
            
            print(f"Loading sentence transformer model: '{self.model_name}'...")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            self._init_milvus_collection()
            print(f"Ensuring collection '{self.collection_name}' is loaded...")
            self.collection.load()

            self._initialized = True
            print("Manager is ready and thread-safe.")

    def _init_milvus_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description="Few-shot instruction retrieval index")

        if not utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' not found. Creating it...")
            self.collection = Collection(name=self.collection_name, schema=schema)
            index_params = {"metric_type": self.metric_type, "index_type": self.index_type, "params": {"M": 16, "efConstruction": 200}}
            self.collection.create_index(field_name="text_embedding", index_params=index_params, index_name="text_index")
            print("Collection and index created successfully.")
        else:
            print(f"Connecting to existing collection: '{self.collection_name}'")
            self.collection = Collection(name=self.collection_name)

    def search_fewshot(self, query: str, top_k: int = 5):
        query_vector = self.model.encode(query)
        
        with self._read_lock(): # This now works correctly
            search_params = {"metric_type": self.metric_type, "params": {"ef": 32}}
            results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="text_embedding",
                param=search_params,
                limit=top_k,
                output_fields=["metadata"]
            )
        
        formatted_results = []
        if results and results[0]:
            for hit in results[0]:
                meta = hit.entity.get("metadata", {})
                formatted_results.append({
                    "query": meta.get("query", ""),
                    "explanation": meta.get("explanation", ""),
                    "visual_checks": meta.get("visual_checks", []),
                    "spatial_instructions": meta.get("spatial_instructions", []),
                    "score": hit.distance
                })
        return formatted_results

    def build_database(self, json_path: str, batch_size: int = 64):
        print("Acquiring exclusive write lock to build database...")
        with self._write_lock(): # This now works correctly
            print("Write lock acquired. Starting database build.")
            if utility.has_collection(self.collection_name):
                print(f"Dropping existing collection '{self.collection_name}'...")
                utility.drop_collection(self.collection_name)
            
            self._init_milvus_collection()
            self.collection.load()

            print(f"Loading data from {json_path}...")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects.")

            batch_texts, batch_metadatas = [], []
            total_inserted = 0

            for i, item in enumerate(data):
                query_text = item.get("query", "").strip()
                if not query_text:
                    continue
                
                batch_texts.append(query_text)
                batch_metadatas.append(item)

                if len(batch_texts) >= batch_size or i == len(data) - 1:
                    if not batch_texts: continue
                    print(f"Processing batch of {len(batch_texts)} items...")
                    
                    text_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                    image_embeddings = [np.zeros(self.embedding_dim).tolist()] * len(batch_texts)
                    text_embeddings_list = text_embeddings.tolist()

                    self.collection.insert([image_embeddings, text_embeddings_list, batch_metadatas])
                    total_inserted += len(batch_texts)
                    print(f"Inserted {len(batch_texts)} records. Total inserted: {total_inserted}")
                    
                    batch_texts, batch_metadatas = [], []

            self.collection.flush()
            print(f"Build complete. Flushed {total_inserted} records to '{self.collection_name}'.")
        print("Write lock released.")
        
    def close(self):
        with self._lock:
            if not self._initialized:
                return
            try:
                print("Releasing collection and disconnecting from Milvus...")
                if hasattr(self, 'collection') and self.collection:
                    self.collection.release()
                connections.disconnect("default")
                self._initialized = False
                MilvusManager._instance = None
            except Exception as e:
                print(f"Error during cleanup: {e}")