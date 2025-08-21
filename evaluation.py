# evaluation.py
import litellm
import json
import numpy as np
from typing import List, Dict
from utils import setup_api_credentials


def evaluate_relevance_with_llm(query: str, item_info: str, config) -> dict:
    """uses an LLM as a judge to evaluate search relevance"""
    model_name = config['models']['evaluation']

    # Provide objective hints to the LLM to guide its decision
    query_words = set(query.lower().split())
    item_words = set(item_info.lower().split())
    common_words = list(query_words.intersection(item_words))

    prompt = f"""
    You are an objective relevance evaluator for a food search engine. 
    Your response MUST be a JSON object with two keys: "score" (an integer from 1 to 3) and "reasoning".
    Rate the relevance on this scale: 3 (Highly Relevant), 2 (Good Relevant), 1 (Irrelevant).

    User Query: "{query}"
    Retrieved Item: "{item_info}"
    Objective Hint: The query and item share these common keywords: {common_words}.
    Based on all information, please provide your score and reasoning.
    """
    try:
        setup_api_credentials(config)
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {"score": 0, "reasoning": "Error during evaluation."}


def calculate_unsupervised_metrics(all_results: List[Dict], config) -> float:
    """calculates the 'confidence gap' as an objective, unsupervised metric"""
    all_gaps = []
    strategy = config['search']['ranking_strategy']

    for result in all_results:
        if len(result['results']) > 1:
            scores = [item['similarity_score'] for item in result['results']]
            top1_score = scores[0]
            avg_of_next_scores = np.mean(scores[1:])

            # Gap calculation depends on whether the score is a distance or similarity
            if strategy == 'euclidean':
                gap = avg_of_next_scores - top1_score  # Lower is better, so gap is inverted
            else:
                gap = top1_score - avg_of_next_scores  # Higher is better

            all_gaps.append(gap)

    return 1 - np.mean(all_gaps) if all_gaps else 0