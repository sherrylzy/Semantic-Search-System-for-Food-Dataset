# main.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
from utils import load_config
from data_processor import process_and_embed_data
from search_engine import SearchEngine
from evaluation import evaluate_relevance_with_llm, calculate_unsupervised_metrics


def run_full_pipeline():
    """Runs the complete search and evaluation pipeline."""
    config = load_config()
    paths = config['paths']

    # Step 1: Process data and generate embeddings (uses caching)
    process_and_embed_data(config)

    # Step 2: Initialize the search engine
    engine = SearchEngine(config)
    if not engine.searcher: return

    # Load necessary data for search and result mapping
    items_df = pd.read_csv(paths['processed_items'])
    queries_df = pd.read_csv(paths['queries_data'])

    model_name = config['search']['embedding_model_in_use']
    sanitized_model_name = model_name.replace("/", "_")
    query_embeddings = np.load(paths['query_embeddings'].replace(".npy", f"_{sanitized_model_name}.npy"))

    all_results = []

    # Step 3: Perform search for all queries
    print("\n--- Step 3: Performing Search for All Queries ---")
    for i, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Executing search"):
        query_text = row['search_term_pt']
        query_vector = query_embeddings[i]

        # Pass all item texts for advanced hybrid scoring
        indices, scores = engine.search(query_text, query_vector)

        top_items = [{
            "rank": rank + 1,
            "itemId": items_df.iloc[item_index]['itemId'],
            "item_info": items_df.iloc[item_index]['combined_text'],
            "similarity_score": float(score)
        } for rank, (item_index, score) in enumerate(zip(indices, scores))]

        all_results.append({"query": query_text, "results": top_items})

    with open("search_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print("\nSearch results saved to search_results.json")

    # Step 4: Run evaluation based on config
    print("\n--- Step 4: Running Final Evaluation ---")

    evaluation_mode = config['settings']['evaluation_mode']
    llm_avg_score = "N/A"
    avg_confidence_gap = "N/A"

    # LLM-as-a-Judge (subjective) evaluation
    if evaluation_mode in ['llm_only', 'full_report']:
        sample_size = config['settings']['llm_judge_sample_size']
        if len(all_results) > sample_size:
            evaluation_sample = random.sample(all_results, sample_size)
        else:
            evaluation_sample = all_results

        evaluation_scores = []
        for result in tqdm(evaluation_sample, desc=f"LLM Random Evaluation (sampling {sample_size})"):
            eval_result = evaluate_relevance_with_llm(
                query=result['query'],
                item_info=result['results'][0]['item_info'] if result['results'] else "No result found",
                config=config
            )
            evaluation_scores.append(eval_result['score'])
        llm_avg_score = np.mean(evaluation_scores) if evaluation_scores else 0

    # Unsupervised metric (objective) evaluation
    if evaluation_mode in ['unsupervised_only', 'full_report']:
        sample_size = config['settings']['unsupervised_sample_size']
        if sample_size == -1 or len(all_results) <= sample_size:
            unsupervised_sample = all_results
        else:
            unsupervised_sample = random.sample(all_results, sample_size)
        avg_confidence_gap = calculate_unsupervised_metrics(unsupervised_sample, config)

    # Final Report
    print("\n" + "=" * 70)
    print(" " * 20 + "Final Unified Evaluation Report")
    print("=" * 70)
    print(f"Configuration:\n"
          f"  - Provider: {config['search']['embedding_provider']}\n"
          f"  - Model: {config['search']['embedding_model_in_use']}\n"
          f"  - Backend: {config['search']['backend']}\n"
          f"  - Ranking: {config['search']['ranking_strategy']}")
    print("-" * 70)
    print("Evaluation Metrics:")
    if llm_avg_score != "N/A":
        sample_count = len(evaluation_sample) if 'evaluation_sample' in locals() else 0
        print(
            f"  - (Subjective) LLM-as-a-Judge Average Score: {llm_avg_score:.2f} / 3.0 ")
    if avg_confidence_gap != "N/A":
        sample_count = len(unsupervised_sample) if 'unsupervised_sample' in locals() else 0
        print(f"  - (Objective) Average Confidence Gap: {avg_confidence_gap:.3f}")
        #print(f"  - (Objective) Average Confidence Gap: {avg_confidence_gap:.4f} (based on {sample_count} queries)")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()