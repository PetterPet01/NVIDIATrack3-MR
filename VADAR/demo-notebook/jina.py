from sentence_transformers import SentenceTransformer, util
import torch
from typing import Tuple

# Load the sentence-transformers model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def get_text_embedding(text: str) -> torch.Tensor:
  embedding = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
  return embedding

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
  return util.cos_sim(a, b).item()

def is_similar_text(text1: str, text2: str, threshold: float = 0.6) -> bool:
  embedding1 = get_text_embedding(text1)
  embedding2 = get_text_embedding(text2)
  similarity = cosine_sim(embedding1, embedding2)
  print(f"Cosine similarity: {similarity}")
  return similarity > threshold

print("SentenceTransformer model loaded.")
while True:
  input_A = input("Text A: ")
  input_B = input("Text B: ")
  print("Similar:", is_similar_text(input_A, input_B))
