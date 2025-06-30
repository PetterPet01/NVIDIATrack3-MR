import os
import time
import getpass
import google.generativeai as genai

# --- 1. CONFIGURE YOUR API KEY ---
# The script will first try to get the API key from an environment variable
# named 'GOOGLE_API_KEY'. If it's not found, it will securely prompt you to enter it.

try:
    # Recommended: Set GOOGLE_API_KEY as an environment variable for security
    api_key = os.environ["GOOGLE_API_KEY"]
except KeyError:
    # Fallback: Prompt for the API key if the environment variable is not set
    print("GOOGLE_API_KEY environment variable not found.")
    api_key = getpass.getpass("Enter your Google AI Studio API Key: ")

genai.configure(api_key=api_key)


# --- 2. SETUP THE MODEL AND PROMPT ---

# We use 'gemini-1.5-flash-latest' which is the fastest model available via the API.
MODEL_NAME = 'gemini-2.5-pro'

# This prompt is designed to generate a moderately long response to get a good speed measurement.
# A short response would be dominated by network latency.
PROMPT = """
Tell me a detailed and imaginative story about a space explorer who discovers 
a planet entirely covered by a single, sentient, crystalline forest. Describe 
the explorer's first landing, their initial interactions with the crystalline 
life forms, and the moment they realize the entire forest shares a single
consciousness. The story should be at least 800 words long.
"""

def run_speed_test():
    """
    Runs the token speed test against the specified Gemini model.
    """
    print("=" * 50)
    print("🚀 Starting Gemini Token Generation Speed Test 🚀")
    print(f"   Model: {MODEL_NAME}")
    print("=" * 50)

    try:
        # Initialize the generative model
        model = genai.GenerativeModel(MODEL_NAME)
        
        print("\n⏳ Sending prompt to the model... (This may take a moment)")

        # Start a high-resolution timer
        start_time = time.perf_counter()

        # Send the prompt and wait for the full response (this is a "unary" call)
        response = model.generate_content(PROMPT)

        # Stop the timer
        end_time = time.perf_counter()

        # --- 3. CALCULATE AND DISPLAY RESULTS ---

        # Calculate the total time taken
        duration = end_time - start_time
        
        # The API response includes usage metadata with token counts
        usage_metadata = response.usage_metadata
        output_tokens = usage_metadata.candidates_token_count
        input_tokens = usage_metadata.prompt_token_count

        if duration > 0:
            # Calculate tokens generated per second
            tokens_per_second = output_tokens / duration
        else:
            tokens_per_second = float('inf') # Should not happen, but for safety

        print("\n✅ Test Complete!")
        print("-" * 30)
        print(f"Total time to generate: {duration:.2f} seconds")
        print(f"Prompt tokens:            {input_tokens}")
        print(f"Generated output tokens:    {output_tokens}")
        print("-" * 30)
        print(f"⚡ Generation Speed: {tokens_per_second:.2f} tokens/second ⚡")
        print("-" * 30)

        # Optional: Print a preview of the response
        print("\n📝 Response Preview:")
        print(response.text[:400] + "...")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check the following:")
        print("  - Is your API key correct and active?")
        print("  - Do you have an internet connection?")
        print("  - Is the model name '{MODEL_NAME}' correct?")


if __name__ == "__main__":
    run_speed_test()