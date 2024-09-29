from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from qdrant_client import QdrantClient
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


def load_qdrant_client(host="qdrant", port="6333"):
    return QdrantClient(url=f"http://{host}:{port}")


client = load_qdrant_client()


def load_documents(input_dir: str):
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
    ).load_data()
    return documents


def load_embedding(model_name="BAAI/bge-base-en-v1.5", method="Fast"):
    if method == "Fast":
        return FastEmbedEmbedding(model_name=model_name)
    else:
        return


def load_vector_store(client, collection_name):
    return QdrantVectorStore(client=client, collection_name=collection_name)


def load_transformations(chunksize=150, chunkoverlap=0):
    return [SentenceSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)]


def upload_documents_qdrant(documents: list, collection_name, client):
    vector_store = load_vector_store(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.embed_model = load_embedding()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=load_transformations(),
    )
    print("Documents uploaded to Qdrant")


if __name__ == "__main__":
    # Ensure the directory matches the one in the Dockerfile
    input_dir = "./data"  # Adjust this path based on your data directory in Docker
    collection_name = "anchor"

    # Load documents and upload them to Qdrant
    documents = load_documents(input_dir)
    client = load_qdrant_client(host="localhost")
    upload_documents_qdrant(documents, collection_name, client)
