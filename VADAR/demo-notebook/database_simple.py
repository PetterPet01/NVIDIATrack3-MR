import numpy as np
import json
import pickle
import os
import time
import threading
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer

class LocalVectorManager:
    """
    A thread-safe, local-only version of MilvusManager that uses NumPy arrays
    for storing and searching text embeddings. It is designed to be a direct
    replacement for the 'nvidia_aic' use case.
    
    Now includes export/import functionality to save and load the vector database
    with full thread safety guarantees.
    """
    def __init__(self, 
                 model_path: str = 'sentence-transformers/paraphrase-MiniLM-L6-v2',
                 database_mode: str = 'nvidia_aic',
                 max_workers: int = 4):
        
        if database_mode != 'nvidia_aic':
            raise ValueError("This simplified manager only supports 'nvidia_aic' mode.")
            
        print(f"Initializing thread-safe local vector manager with model: {model_path}")
        
        # Thread safety locks
        self._data_lock = threading.RLock()  # Recursive lock for nested operations
        self._model_lock = threading.Lock()   # Lock for model operations
        self._io_lock = threading.Lock()      # Lock for file I/O operations
        
        # Model initialization (thread-safe)
        with self._model_lock:
            self.model = SentenceTransformer(model_path)
            self.model_path = model_path
        
        # Local in-memory storage for embeddings and metadata (protected by _data_lock)
        self._embeddings = None
        self._metadata = []
        
        # Thread pool for parallel operations
        self._max_workers = max_workers
        self._thread_pool = None
        
        # Database state tracking
        self._is_built = threading.Event()
        self._build_lock = threading.Lock()  # Ensures only one build operation at a time

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Thread-safe getter for embeddings."""
        with self._data_lock:
            return self._embeddings.copy() if self._embeddings is not None else None

    @property
    def metadata(self) -> List[Dict]:
        """Thread-safe getter for metadata."""
        with self._data_lock:
            return copy.deepcopy(self._metadata)

    @contextmanager
    def _get_thread_pool(self):
        """Context manager for thread pool to ensure proper cleanup."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=self._max_workers)
        try:
            yield self._thread_pool
        finally:
            pass  # Keep pool alive for reuse

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()

    def cleanup(self):
        """Clean up thread pool and other resources."""
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None

    def build(self, data_root: str, mode: str):
        """
        Thread-safe method to build the local vector database from a data source.
        'data_root' is expected to be the path to the JSON file.
        """
        with self._build_lock:  # Ensure only one build operation at a time
            if mode == 'nvidia_aic':
                self.__build_nvidia_aic(data_root)
            else:
                raise ValueError(f"Build mode '{mode}' is not supported by this simplified manager.")

    def __build_nvidia_aic(self, json_path: str):
        """
        Thread-safe method to load data from a JSON file, encode the 'query' field, 
        and store the embeddings and metadata in memory.
        """
        print(f"Building local database from: {json_path}")
        
        # File I/O operations are protected
        with self._io_lock:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"Error: The file was not found at {json_path}")
                return
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from the file {json_path}")
                return

        if not isinstance(data, list):
            raise ValueError("Expected a list of JSON objects in the input file.")

        # Extract queries to encode and the full metadata to store
        queries_to_encode = [item.get("query", "").strip() for item in data if item.get("query", "").strip()]
        metadata_to_store = [item for item in data if item.get("query", "").strip()]

        if not queries_to_encode:
            print("No valid queries found in the JSON file to build the database.")
            return

        print(f"Found {len(queries_to_encode)} queries. Encoding now...")
        
        # Model encoding is thread-safe with model lock
        with self._model_lock:
            embeddings = self.model.encode(queries_to_encode, show_progress_bar=True)
        
        # Update internal state atomically
        with self._data_lock:
            self._embeddings = embeddings
            self._metadata = metadata_to_store
        
        # Signal that database is built
        self._is_built.set()
        
        print(f"Successfully built local database with {len(metadata_to_store)} entries.")

    def export_database(self, export_path: str):
        """
        Thread-safe method to export the current vector database to a file.
        
        Args:
            export_path (str): Path where to save the database file (should end with .pkl)
        """
        # Wait for database to be built
        if not self._is_built.wait(timeout=300):  # 5 minute timeout
            raise RuntimeError("Database build operation timed out.")
        
        # Create export data with thread-safe access
        with self._data_lock:
            if self._embeddings is None or not self._metadata:
                raise RuntimeError("No database to export. Build the database first using .build() method.")
            
            export_data = {
                'embeddings': self._embeddings.copy(),
                'metadata': copy.deepcopy(self._metadata),
                'model_path': self.model_path
            }
        
        # File I/O is protected
        with self._io_lock:
            try:
                with open(export_path, 'wb') as f:
                    pickle.dump(export_data, f)
                print(f"Database successfully exported to: {export_path}")
                print(f"Exported {len(export_data['metadata'])} entries with embeddings of shape {export_data['embeddings'].shape}")
            except Exception as e:
                print(f"Error exporting database: {e}")
                raise

    def load_database(self, import_path: str):
        """
        Thread-safe method to load a previously exported vector database from a file.
        
        Args:
            import_path (str): Path to the exported database file
        """
        with self._build_lock:  # Treat load as a build operation
            if not os.path.exists(import_path):
                raise FileNotFoundError(f"Database file not found at: {import_path}")
            
            # File I/O is protected
            with self._io_lock:
                try:
                    with open(import_path, 'rb') as f:
                        export_data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading database: {e}")
                    raise
            
            # Verify the exported model path matches current model
            exported_model_path = export_data.get('model_path', '')
            if exported_model_path != self.model_path:
                print(f"Warning: Exported model path '{exported_model_path}' doesn't match current model '{self.model_path}'")
                print("This may cause inconsistent results. Consider rebuilding the database.")
            
            # Update internal state atomically
            with self._data_lock:
                self._embeddings = export_data['embeddings']
                self._metadata = export_data['metadata']
            
            # Signal that database is loaded
            self._is_built.set()
            
            print(f"Database successfully loaded from: {import_path}")
            print(f"Loaded {len(export_data['metadata'])} entries with embeddings of shape {export_data['embeddings'].shape}")

    def search_fewshot(self, query: str, top_k: int = 5, verbose: bool = False) -> List[Dict]:
        """
        Thread-safe search method for the in-memory database using cosine similarity.

        Args:
            query (str): User query string.
            top_k (int): Number of similar queries to retrieve.
            verbose (bool): If True, prints timing information for each step.

        Returns:
            List[Dict]: A list of the top_k most similar entries, formatted as specified.
        """
        # Wait for database to be ready
        if not self._is_built.wait(timeout=30):  # 30 second timeout for search
            raise RuntimeError("Database is not ready. Call the .build() method or .load_database() method first.")

        total_start = time.time()
        
        # 1. Encode the search query into a vector (thread-safe)
        encode_start = time.time()
        with self._model_lock:
            query_vector = self.model.encode(query)
        encode_time = time.time() - encode_start
        
        # 2. Get thread-safe snapshots of data for computation
        similarity_start = time.time()
        with self._data_lock:
            if self._embeddings is None or not self._metadata:
                raise RuntimeError("The database has not been built yet.")
            
            # Create local copies for computation
            embeddings_copy = self._embeddings.copy()
            metadata_copy = copy.deepcopy(self._metadata)
        
        # 3. Compute cosine similarity (no locks needed for computation)
        scores = np.dot(embeddings_copy, query_vector)
        similarity_time = time.time() - similarity_start
        
        # 4. Get the indices of the top_k highest scores
        sort_start = time.time()
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        sort_time = time.time() - sort_start
        
        # 5. Format the results
        format_start = time.time()
        formatted_results = []
        for index in top_k_indices:
            meta = metadata_copy[index]
            score = scores[index]
            formatted_results.append({
                "query": meta.get("query", ""),
                "explanation": meta.get("explanation", ""),
                "visual_checks": meta.get("visual_checks", []),
                "spatial_instructions": meta.get("spatial_instructions", []),
                "score": float(score)  # Convert NumPy float to standard Python float
            })
        format_time = time.time() - format_start
        
        total_time = time.time() - total_start
        
        # Store timing information in the results
        timing_info = {
            "encoding_time": encode_time,
            "similarity_computation_time": similarity_time,
            "sorting_time": sort_time,
            "formatting_time": format_time,
            "total_search_time": total_time
        }
        
        if verbose:
            print(f"\n--- Search Timing Breakdown ---")
            print(f"Query encoding: {encode_time:.4f}s")
            print(f"Similarity computation: {similarity_time:.4f}s")
            print(f"Result sorting: {sort_time:.4f}s")
            print(f"Result formatting: {format_time:.4f}s")
            print(f"Total search time: {total_time:.4f}s")
            print(f"Database size: {len(metadata_copy)} entries")
        
        # Add timing info to the first result for easy access
        if formatted_results:
            formatted_results[0]["timing_info"] = timing_info
            
        return formatted_results

    def search_parallel(self, queries: List[str], top_k: int = 5, max_workers: Optional[int] = None) -> List[List[Dict]]:
        """
        Perform parallel searches for multiple queries.
        
        Args:
            queries (List[str]): List of query strings to search
            top_k (int): Number of results per query
            max_workers (Optional[int]): Number of worker threads (defaults to instance setting)
            
        Returns:
            List[List[Dict]]: Results for each query in the same order
        """
        if not queries:
            return []
            
        workers = max_workers or self._max_workers
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all search tasks
            future_to_index = {
                executor.submit(self.search_fewshot, query, top_k): i 
                for i, query in enumerate(queries)
            }
            
            # Collect results in order
            results = [None] * len(queries)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Error processing query {index}: {e}")
                    results[index] = []
                    
        return results

    def ensure_collection_loaded(self):
        """
        Thread-safe dummy method for API compatibility with the original MilvusManager.
        Waits for the database to be ready.
        """
        if self._is_built.wait(timeout=30):
            print("Collection is already in memory, no loading action needed.")
        else:
            print("Warning: Database is not ready yet.")

    def benchmark_search(self, query: str, num_runs: int = 10, top_k: int = 5, 
                        parallel: bool = False, max_workers: Optional[int] = None) -> Dict:
        """
        Thread-safe benchmark method that runs multiple search queries to benchmark performance.
        
        Args:
            query (str): Query to benchmark
            num_runs (int): Number of times to run the search
            top_k (int): Number of results to retrieve
            parallel (bool): Whether to run searches in parallel
            max_workers (Optional[int]): Number of worker threads for parallel execution
            
        Returns:
            Dict: Statistics including mean, min, max times for each operation
        """
        if not self._is_built.wait(timeout=30):
            raise RuntimeError("The database has not been built yet.")

        print(f"Running {num_runs} search iterations for benchmarking...")
        
        if parallel:
            # Parallel benchmark
            queries = [query] * num_runs
            start_time = time.time()
            results = self.search_parallel(queries, top_k, max_workers)
            total_time = time.time() - start_time
            
            # Calculate average per-query time
            avg_time = total_time / num_runs
            
            benchmark_results = {
                "num_runs": num_runs,
                "database_size": len(self._metadata),
                "top_k": top_k,
                "parallel": True,
                "max_workers": max_workers or self._max_workers,
                "total_time": total_time,
                "avg_time_per_query": avg_time,
                "queries_per_second": num_runs / total_time
            }
            
            print(f"\n--- Parallel Benchmark Results ({num_runs} runs) ---")
            print(f"Total time: {total_time:.4f}s")
            print(f"Average time per query: {avg_time:.4f}s")
            print(f"Queries per second: {benchmark_results['queries_per_second']:.2f}")
            
        else:
            # Sequential benchmark (original implementation)
            encoding_times = []
            similarity_times = []
            sorting_times = []
            formatting_times = []
            total_times = []
            
            for i in range(num_runs):
                start_total = time.time()
                
                # Encoding
                start_encode = time.time()
                with self._model_lock:
                    query_vector = self.model.encode(query)
                encoding_times.append(time.time() - start_encode)
                
                # Get data snapshot
                with self._data_lock:
                    embeddings_copy = self._embeddings.copy()
                    metadata_copy = copy.deepcopy(self._metadata)
                
                # Similarity computation
                start_sim = time.time()
                scores = np.dot(embeddings_copy, query_vector)
                similarity_times.append(time.time() - start_sim)
                
                # Sorting
                start_sort = time.time()
                top_k_indices = np.argsort(scores)[-top_k:][::-1]
                sorting_times.append(time.time() - start_sort)
                
                # Formatting
                start_format = time.time()
                formatted_results = []
                for index in top_k_indices:
                    meta = metadata_copy[index]
                    score = scores[index]
                    formatted_results.append({
                        "query": meta.get("query", ""),
                        "explanation": meta.get("explanation", ""),
                        "visual_checks": meta.get("visual_checks", []),
                        "spatial_instructions": meta.get("spatial_instructions", []),
                        "score": float(score)
                    })
                formatting_times.append(time.time() - start_format)
                
                total_times.append(time.time() - start_total)
            
            # Calculate statistics
            def calc_stats(times):
                return {
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "median": np.median(times)
                }
            
            with self._data_lock:
                db_size = len(self._metadata)
            
            benchmark_results = {
                "num_runs": num_runs,
                "database_size": db_size,
                "top_k": top_k,
                "parallel": False,
                "encoding": calc_stats(encoding_times),
                "similarity_computation": calc_stats(similarity_times),
                "sorting": calc_stats(sorting_times),
                "formatting": calc_stats(formatting_times),
                "total": calc_stats(total_times)
            }
            
            # Print summary
            print(f"\n--- Sequential Benchmark Results ({num_runs} runs) ---")
            print(f"Database size: {db_size} entries")
            print(f"Query encoding: {benchmark_results['encoding']['mean']:.4f}s ± {benchmark_results['encoding']['std']:.4f}s")
            print(f"Similarity computation: {benchmark_results['similarity_computation']['mean']:.4f}s ± {benchmark_results['similarity_computation']['std']:.4f}s")
            print(f"Result sorting: {benchmark_results['sorting']['mean']:.4f}s ± {benchmark_results['sorting']['std']:.4f}s")
            print(f"Result formatting: {benchmark_results['formatting']['mean']:.4f}s ± {benchmark_results['formatting']['std']:.4f}s")
            print(f"Total search time: {benchmark_results['total']['mean']:.4f}s ± {benchmark_results['total']['std']:.4f}s")
            print(f"Searches per second: {1/benchmark_results['total']['mean']:.2f}")
        
        return benchmark_results

    def get_database_info(self) -> Dict[str, Any]:
        """
        Thread-safe method to get information about the current database state.
        
        Returns:
            Dict: Information about the database
        """
        with self._data_lock:
            if self._embeddings is None:
                return {
                    "status": "not_built",
                    "entries": 0,
                    "embedding_shape": None,
                    "model_path": self.model_path,
                    "is_ready": self._is_built.is_set()
                }
            else:
                return {
                    "status": "ready",
                    "entries": len(self._metadata),
                    "embedding_shape": self._embeddings.shape,
                    "model_path": self.model_path,
                    "is_ready": self._is_built.is_set()
                }


# if __name__ == '__main__':
#     with LocalVectorManager(database_mode='nvidia_aic', max_workers=4) as manager:
    
#         # Option 1: Build from scratch and export
#         print("=== Building and Exporting Database ===")
#         # Uncomment the following lines if you have the JSON file
#         manager.build(data_root='/workspace/MealsretrivevalDatabase/batch_4_refined.json', mode='nvidia_aic')
#         manager.export_database('my_vector_database.pkl')

# Example usage demonstrating thread-safe operations
if __name__ == '__main__':
    import threading
    import time
    
    def test_concurrent_searches(manager, query, num_searches=10):
        """Test function for concurrent search operations."""
        print(f"Thread {threading.current_thread().name}: Starting {num_searches} searches")
        for i in range(num_searches):
            results = manager.search_fewshot(query, top_k=3)
            print(f"Thread {threading.current_thread().name}: Search {i+1} completed, found {len(results)} results")
        print(f"Thread {threading.current_thread().name}: All searches completed")
    
    # Create manager with context management
    with LocalVectorManager(database_mode='nvidia_aic', max_workers=4) as manager:
        
        # Option 1: Build from scratch and export
        print("=== Building and Exporting Database ===")
        # Uncomment the following lines if you have the JSON file
        manager.build(data_root='/workspace/MealsretrivevalDatabase/batch_4_refined.json', mode='nvidia_aic')
        manager.export_database('my_vector_database.pkl')
        
        # Option 2: Load from exported file (uncomment if you have the file)
        print("=== Loading from Exported Database ===")
        manager.load_database('my_vector_database.pkl')
        
        # For demonstration, let's create a simple test database
        print("=== Creating Test Database ===")
        test_data = [
            {"query": "What is machine learning?", "explanation": "ML is AI subset", "visual_checks": [], "spatial_instructions": []},
            {"query": "How does neural network work?", "explanation": "Networks process data", "visual_checks": [], "spatial_instructions": []},
            {"query": "What is deep learning?", "explanation": "Deep learning uses neural networks", "visual_checks": [], "spatial_instructions": []}
        ]
        
        # Create a temporary JSON file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(test_data, tmp_file)
            tmp_file_path = tmp_file.name
        
        try:
            manager.build(data_root=tmp_file_path, mode='nvidia_aic')
            
            # Test basic search
            query = 'What is machine learning?'
            results = manager.search_fewshot(query, verbose=True)
            
            print("\n--- Search Results ---")
            for result in results:
                print(f"Query: {result['query']}")
                print(f"Score: {result['score']:.4f}\n")
            
            # Test concurrent searches
            print("\n=== Testing Thread Safety ===")
            threads = []
            for i in range(3):
                thread = threading.Thread(
                    target=test_concurrent_searches, 
                    args=(manager, query, 5),
                    name=f"SearchThread-{i+1}"
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Test parallel search
            print("\n=== Testing Parallel Search ===")
            queries = [query] * 5
            parallel_results = manager.search_parallel(queries, top_k=2)
            print(f"Parallel search completed for {len(parallel_results)} queries")
            
            # Test benchmark
            print("\n=== Running Benchmark ===")
            benchmark_results = manager.benchmark_search(query, num_runs=20)
            
            # Test parallel benchmark
            print("\n=== Running Parallel Benchmark ===")
            parallel_benchmark = manager.benchmark_search(query, num_runs=20, parallel=True, max_workers=3)
            
            # Show database info
            print("\n--- Database Info ---")
            info = manager.get_database_info()
            print(f"Status: {info['status']}")
            print(f"Entries: {info['entries']}")
            print(f"Embedding Shape: {info['embedding_shape']}")
            print(f"Model Path: {info['model_path']}")
            print(f"Is Ready: {info['is_ready']}")
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    print("\n=== All tests completed successfully! ===")


