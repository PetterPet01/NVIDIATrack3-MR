from sentence_transformers import SentenceTransformer, util
# Load a pre-trained embedding model (only once)
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', trust_remote_code=True)

from typing import List
from sklearn.metrics.pairwise import cosine_similarity

def is_similar_text(text1: str, text2: str) -> bool:
    """
    Determine whether two texts are semantically similar using cosine similarity.

    Args:
        text1 (str): First text input.
        text2 (str): Second text input.

    Returns:
        bool: True if similarity is above threshold, False otherwise.
    """
    threshold = 0.6
    prompt_embedding = embedding_model.encode([text1], convert_to_numpy=True)
    class_embedding = embedding_model.encode([text2], convert_to_numpy=True)

    similarity = cosine_similarity(prompt_embedding, class_embedding)[0][0]
    print(f"Cosine similarity: {similarity}")
    return similarity > threshold

print('loaded')
while True:
    input_A = input()
    input_B = input()
    print(is_similar_text(input_A, input_B))
    