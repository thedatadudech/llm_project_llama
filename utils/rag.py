import os
from dotenv import load_dotenv
from data_processing.qdrant_loader import load_qdrant_client, load_embedding
from utils.ground_truth import extract_text_from_node_content
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")


client = OpenAI(api_key=api_key)
qdrant_client = load_qdrant_client(host="qdrant")
embedding_model = load_embedding()


def qdrant_search(query, embedding_model, collection_name, query_filter):
    query_vector = embedding_model.get_text_embedding(query)
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,  # Provide the query vector here
        limit=5,
    )
    result_docs = []
    search_scoring = []
    for res in search_results:
        retrieved_answer = extract_text_from_node_content(res.payload)
        result_docs.append(retrieved_answer)
        search_scoring.append(res.score)
    return result_docs


def build_prompt(query, search_results):
    prompt_template = """
You're a anchor expert. Answer the QUESTION based on the CONTEXT fro.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""

    for doc in search_results:
        context = context + doc

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def rag(query, source):
    filter_conditions = None

    if source:
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="file_name",  # The metadata field you want to filter on
                    match=MatchValue(
                        value=source
                    ),  # The value to match in the 'category' field
                ),
            ]
        )

    search_results = qdrant_search(
        query,
        embedding_model=embedding_model,
        collection_name="anchor",
        query_filter=filter_conditions,
    )
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
