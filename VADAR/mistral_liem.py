import asyncio
import os
import random
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from mistralai import Mistral
import pandas as pd

# --- Configuration ---
GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1NAlj7OiD9apH3U47RLJK0en1wLSW78X5zqmf6NmVUA4/export?format=csv&gid=0"
)


# --- Helper Function to Load Keys ---
def load_keys_from_google_sheet(url: str) -> list[str]:
    """
    Loads API keys from the first column of a public Google Sheet (CSV-export link),
    skipping the first row and filtering out blank/invalid entries.
    """
    print("🔄 Loading API Keys from Google Sheet...")
    try:
        df = pd.read_csv(
            url,
            skiprows=1,       # Skip the first row (header)
            on_bad_lines='skip',
            encoding='utf-8',
            usecols=[0],      # First column only
            header=None       # Prevent pandas from assuming first row is header
        )

        # Clean and filter keys
        api_keys = df[0].dropna().astype(str).map(str.strip)
        valid_keys = [key for key in api_keys if key]

        if not valid_keys:
            print("⚠️ No valid API keys found.")
            return []

        print(f"✅ Loaded {len(valid_keys)} API keys.")
        return valid_keys

    except Exception as e:
        print(f"❌ Error loading API keys: {e}")
        return []

ALL_API_KEYS = load_keys_from_google_sheet(GOOGLE_SHEET_URL)

# --- Core Logic ---

async def process_single_request(
    prompt: str, 
    key_queue: asyncio.Queue, 
    session_id: int
) -> Dict[str, Any]:
    """
    Worker function to process one API request.
    It gets a key from the queue, makes the call, and returns the key.
    """
    api_key = None
    start_time = time.time()
    
    try:
        # 1. "Check out" an API key from the pool. This will wait if no key is available.
        api_key = await key_queue.get()
        print(f"[Session {session_id}] Acquired key: ...{api_key[-4:]}")

        # 2. Initialize the client with the specific key
        client = Mistral(api_key=api_key)

        # 3. Make the API call using the newer client structure
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Use asyncio.to_thread to run the sync client in an async context
        chat_response = await asyncio.to_thread(
            client.chat.complete,
            model="mistral-medium-2505",  # Or your preferred model
            messages=messages
        )

        end_time = time.time()
        duration = end_time - start_time

        # 4. Extract token usage information
        response_content = chat_response.choices[0].message.content
        
        # Get token usage from the response
        usage = chat_response.usage if hasattr(chat_response, 'usage') else None
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        
        print(f"[Session {session_id}] Request successful. Key ...{api_key[-4:]} will be returned to pool.")
        return {
            "status": "success",
            "prompt": prompt,
            "response": response_content,
            "key_used": f"...{api_key[-4:]}", # For logging/verification
            "duration": duration,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": total_tokens / duration if duration > 0 else 0
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[Session {session_id}] Request failed with key ...{api_key[-4:] if api_key else 'N/A'}. Error: {e}")
        return {
            "status": "error",
            "prompt": prompt,
            "error_message": str(e),
            "key_used": f"...{api_key[-4:]}" if api_key else "N/A",
            "duration": duration,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_per_second": 0
        }
    finally:
        # 5. CRITICAL: Return the key to the queue so other tasks can use it.
        # This block runs whether the try block succeeded or failed.
        if api_key:
            await key_queue.put(api_key)
            # print(f"[Session {session_id}] Returned key: ...{api_key[-4:]} to the pool.")

async def run_concurrent_mistral_requests(
    prompts: List[str],
    all_api_keys: List[str],
    n_concurrent: int
) -> List[Dict[str, Any]]:
    """
    Runs Mistral API requests concurrently using a pool of distinct API keys.

    Args:
        prompts: A list of prompts to send to the API.
        all_api_keys: The full list of available Mistral API keys.
        n_concurrent: The number of concurrent requests to run (N).

    Returns:
        A list of dictionaries, each containing the result of a request.
    """
    if n_concurrent > len(all_api_keys):
        raise ValueError(f"Concurrency level (N={n_concurrent}) cannot be greater than the number of available API keys ({len(all_api_keys)}).")
    
    # Create a queue that will act as our API key pool
    key_queue = asyncio.Queue()

    # Add N distinct keys to the queue. These are the only keys that will be used.
    # We take a slice of the first N keys.
    keys_for_pool = all_api_keys[:n_concurrent]
    for key in keys_for_pool:
        await key_queue.put(key)

    print(f"--- Starting concurrent processing ---")
    print(f"Total prompts to process: {len(prompts)}")
    print(f"Concurrency level (N): {n_concurrent}")
    print(f"API Key pool size: {key_queue.qsize()}")
    print("-" * 35)

    # Record start time for overall throughput calculation
    overall_start_time = time.time()

    # Create a list of tasks to run concurrently
    tasks = [
        process_single_request(prompt, key_queue, i)
        for i, prompt in enumerate(prompts)
    ]

    # Run all tasks and wait for them to complete
    results = await asyncio.gather(*tasks)
    
    # Record end time
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    
    # Calculate throughput metrics
    total_tokens = sum(result.get('total_tokens', 0) for result in results)
    successful_requests = sum(1 for result in results if result['status'] == 'success')
    
    # Add overall metrics to results
    throughput_info = {
        'total_duration': total_duration,
        'total_tokens': total_tokens,
        'overall_tokens_per_second': total_tokens / total_duration if total_duration > 0 else 0,
        'successful_requests': successful_requests,
        'total_requests': len(results)
    }
    
    return results, throughput_info

# --- Example Usage ---

async def main():
    """Main function to demonstrate the concurrent runner."""
    
    # Example: We have 30 prompts to process
    sample_prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
        "Write a short poem about the moon.",
        "What are the main benefits of using Python for data science?",
        "Summarize the plot of the movie Inception.",
        "What is quantum computing?",
        "Give me a recipe for chocolate chip cookies.",
        "Who was Leonardo da Vinci?",
        "Explain the difference between TCP and UDP.",
        "Write a python function to find the factorial of a number.",
        "What is the capital of Japan?",
        "Explain machine learning in one paragraph.",
        "Write a haiku about coding.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "What is blockchain technology?",
        "Give me tips for better sleep hygiene.",
        "Who invented the telephone?",
        "Explain the difference between HTTP and HTTPS.",
        "Write a function to reverse a string in Python.",
        "What is the largest planet in our solar system?",
        "Explain artificial intelligence briefly.",
        "Write a limerick about programming.",
        "What are the advantages of cloud computing?",
        "Describe how vaccines work.",
        "What is cryptocurrency?",
        "Give me healthy breakfast ideas.",
        "Who painted the Mona Lisa?",
        "Explain the difference between SQL and NoSQL.",
        "Write a Python function to check if a number is prime.",
    ]
    
    # We want to run N=20 requests concurrently
    N_CONCURRENT_REQUESTS = 20

    # Ensure we have enough keys for the demo
    if len(ALL_API_KEYS) < N_CONCURRENT_REQUESTS:
        print(f"Warning: You only provided {len(ALL_API_KEYS)} keys, but N is {N_CONCURRENT_REQUESTS}. Using all available keys.")
        N_CONCURRENT_REQUESTS = len(ALL_API_KEYS)

    if N_CONCURRENT_REQUESTS == 0:
        print("No API keys found. Exiting.")
        return

    # Run the concurrent requests
    final_results, throughput_info = await run_concurrent_mistral_requests(
        prompts=sample_prompts,
        all_api_keys=ALL_API_KEYS,
        n_concurrent=N_CONCURRENT_REQUESTS
    )

    print("\n--- All requests completed ---")
    successful_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    for i, result in enumerate(final_results):
        print(f"\nResult for Prompt #{i+1}: '{result['prompt'][:50]}...'")
        print(f"  Status: {result['status']}")
        print(f"  Key Used (last 4 digits): {result['key_used']}")
        print(f"  Duration: {result['duration']:.2f}s")
        
        if result['status'] == 'success':
            successful_count += 1
            total_prompt_tokens += result['prompt_tokens']
            total_completion_tokens += result['completion_tokens']
            print(f"  Tokens: {result['total_tokens']} (prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']})")
            print(f"  Individual throughput: {result['tokens_per_second']:.2f} tokens/sec")
            print(f"  Response: {result['response'][:100]}...") # Print first 100 chars
        else:
            print(f"  Error: {result['error_message']}")
    
    # Print comprehensive throughput summary        
    print(f"\n{'='*60}")
    print(f"THROUGHPUT SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {throughput_info['total_duration']:.2f} seconds")
    print(f"Successful requests: {throughput_info['successful_requests']}/{throughput_info['total_requests']}")
    print(f"Total tokens processed: {throughput_info['total_tokens']:,}")
    print(f"  - Prompt tokens: {total_prompt_tokens:,}")
    print(f"  - Completion tokens: {total_completion_tokens:,}")
    print(f"")
    print(f"🚀 OVERALL THROUGHPUT: {throughput_info['overall_tokens_per_second']:.2f} tokens/second")
    print(f"Average tokens per successful request: {throughput_info['total_tokens']/throughput_info['successful_requests']:.1f}" if throughput_info['successful_requests'] > 0 else "N/A")
    print(f"Requests per second: {throughput_info['successful_requests']/throughput_info['total_duration']:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())