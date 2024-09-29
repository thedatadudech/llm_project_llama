import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import openai

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=api_key)
import numpy as np
import json


def load_ground_truth(ground_truth_file: str) -> None:
    """Loads the ground truth data from a JSON file."""
    with open(ground_truth_file, "r") as f:
        return json.load(f)


def compute_similarity(text1: str, text2: str):
    model_c = SentenceTransformer(
        "all-MiniLM-L6-v2", config_kwargs={"clean_up_tokenization_spaces": True}
    )
    vector1 = model_c.encode([text1])[0]
    vector2 = model_c.encode([text2])[0]
    similarity = cosine_similarity([vector1], [vector2])
    return similarity[0][0]


def extract_text_from_node_content(payload) -> None:
    if "_node_content" in payload:
        # Parse the JSON string stored in _node_content
        node_content = json.loads(payload["_node_content"])
        if "text" in node_content:
            return node_content["text"]
    return None


def rate_relevance(
    question: str,
    true_answer: str,
    retrieved_answer: str,
    llm_model="gpt-4o",
) -> float:
    prompt = f"""
    Question: {question}
    True Answer: {true_answer}
    Retrieved Answer: {retrieved_answer}
    Rate the relevance of the retrieved answer to the question on a scale of 1 (not relevant) to 5 (high relevant).
    Return only the number.
    """
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are an expert at evaluating answers."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
        temperature=0.0,  # Set to 0 for deterministic output
    )
    rating_str = response.choices[0].message.content
    try:
        rating = float(rating_str)
    except ValueError:
        rating = None
    return rating
