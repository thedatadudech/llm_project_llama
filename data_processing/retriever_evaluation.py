from utils.ground_truth import (
    extract_text_from_node_content,
    compute_similarity,
    rate_relevance,
)


def rank_rerank(
    search_results, question, expected_answer, llm_model, top=1, re_rank=False
):
    rank_reranked = []
    for res in search_results:
        retrieved_answer = extract_text_from_node_content(res.payload)
        similarity = compute_similarity(expected_answer, retrieved_answer)
        rating = rate_relevance(
            question=question,
            true_answer=expected_answer,
            retrieved_answer=retrieved_answer,
            llm_model=llm_model,
        )
        combined_score = similarity * 0.7 + rating * 0.3
        rank_reranked.append((retrieved_answer, similarity, rating, combined_score))
    if re_rank:
        rank_reranked.sort(key=lambda x: x[3], reverse=True)

    return rank_reranked[:top]


def evaluate_search_results(
    ground_truth,
    qdrant_client,
    embedding_model,
    collection_name="anchor",
    llm_model="gpt-4o-mini",
    similarity_threshold=0.7,
    n_search_results=1,
    reranking=False,
    verbose=True,
):
    total_questions = len(ground_truth)
    hits = []
    ratings = []
    combined_list = []

    for element in ground_truth:
        question = element["question"]
        expected_answer = element["answer"]
        hit = False
        model = embedding_model
        query_vector = model.get_text_embedding(question)
        print(f"Question: {question}") if verbose else ""
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,  # Provide the query vector here
            limit=n_search_results,
        )

        if search_results:
            retrieved_answer, similarity, rating, combined = rank_rerank(
                search_results=search_results,
                question=question,
                expected_answer=expected_answer,
                llm_model=llm_model,
                re_rank=reranking,
            )[0]
            if verbose:
                print(f"Expected: {expected_answer}")
                print(f"Retrieved: {retrieved_answer}")
                print(f"Similarity: {similarity:.2f}")
                print(f"Rating: {rating}")
                print(f"Combined: {combined}")

            if similarity >= similarity_threshold:
                hit = True
                hits.append(hit)
                ratings.append(rating)
                combined_list.append(combined)
                print("Match: ✅") if verbose else ""
                continue
            else:
                print("Match: ❌") if verbose else ""

            ratings.append(rating)
            combined_list.append(combined)

        else:
            print("No results found") if verbose else ""

    hit_rate = sum(hits) / total_questions
    rating_avg = sum(ratings) / total_questions
    combined_avg = sum(combined_list) / total_questions

    print("Ratings", rating) if verbose else ""
    print(f"Hit_Rate: {hit_rate * 100:.2f}%")
    print(f"Average Rating: {rating_avg} / 5")
    print(f"Average Combined Rating: {combined_avg} / 2.2")
    return hit_rate, rating_avg, ratings, combined_list
