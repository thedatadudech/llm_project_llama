import json


def create_hash_map_from_documents(documents: list, hash_map: dict):
    for document in documents:
        document_hash = document.hash
        hash_map[document_hash] = document.id_
    return hash_map


def load_hash_map(filename="hash_map.json"):
    try:
        with open(filename, "r") as f:
            hash_map = json.load(f)
        print(f"Hash map loaded from {filename}")
        return hash_map
    except FileNotFoundError:
        print(f"{filename} not found. Starting with an empty hash map.")
        return {}
