import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import litellm
from utils import load_config, setup_api_credentials
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Global cache for the local model to avoid reloading
local_model_cache: Optional[SentenceTransformer] = None
local_model_name_cache: Optional[str] = None


def get_embeddings_from_api(texts: List[str], model: str, dimensions: int, batch_size: int = 50) -> np.ndarray:
    """fetches embeddings from the API in batches."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"API Batch Generation (size={batch_size})"):
        batch = [str(t) for t in texts[i:i + batch_size]]
        try:
            response = litellm.embedding(model=model, input=batch)
            embeddings = [item['embedding'] for item in response.data]
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"Error in batch {i // batch_size}: {e}")
            all_embeddings.extend([[0.0] * dimensions] * len(batch))
    return np.array(all_embeddings)


def get_embeddings_from_local(texts: List[str], model_name: str) -> np.ndarray:
    """generates embeddings locally using a hugge-face model."""
    global local_model_cache, local_model_name_cache
    if local_model_name_cache != model_name:
        print(f"Loading local model from Hugging Face: {model_name}...")
        local_model_cache = SentenceTransformer(model_name)
        local_model_name_cache = model_name
    return local_model_cache.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def preprocess_items(df: pd.DataFrame) -> pd.DataFrame:
    """creates a structured and information-rich text document for each item."""

    def combine_text(row):
        try:
            metadata = json.loads(row['itemMetadata'])
            taxonomy = metadata.get('taxonomy', {})
            name = metadata.get('name', '')
            description = metadata.get('description', '')
            category = metadata.get('category_name', '')
            l0 = taxonomy.get('l0', '')
            l1 = taxonomy.get('l1', '')

            # Feature Engineering: Create a structured document
            structured_text = (
                f"Item Name: {name}. "
                f"Category: {category}, {l0}, {l1}. "
                f"Description: {description}. "
                f"Keywords for searching: {name}, {category}."
            )
            return structured_text
        except (json.JSONDecodeError, TypeError):
            return ""

    df['combined_text'] = df.apply(combine_text, axis=1)
    return df


def process_and_embed_data(config):
    """handle data preprocessing and embedding generation."""
    print("\n--- Step 1: Processing Data and Generating Embeddings ---")

    provider = config['search']['embedding_provider']
    model_name = config['search']['embedding_model_in_use']

    # determine model dimensions from the correct config section
    if provider == 'api':
        setup_api_credentials(config)
        dimensions = config['models']['api_embedding_options'][model_name]
    elif provider == 'local':
        dimensions = config['models']['local_embedding_options'][model_name]
    else:
        raise ValueError(f"Unknown provider in config: {provider}")

    paths = config['paths']
    os.makedirs(paths['output_dir'], exist_ok=True)

    print(f"Provider: {provider}, Model: {model_name} (Dimensions: {dimensions})")

    # model-specific filenames to avoid conflicts
    sanitized_model_name = model_name.replace("/", "_")
    item_emb_path = paths['item_embeddings'].replace(".npy", f"_{sanitized_model_name}.npy")
    query_emb_path = paths['query_embeddings'].replace(".npy", f"_{sanitized_model_name}.npy")

    # embed items
    if not os.path.exists(item_emb_path) or not os.path.exists(paths['processed_items']):
        print("\nProcessing item data...")
        items_df = pd.read_csv(paths['items_data'])
        items_df = preprocess_items(items_df)
        items_df.to_csv(paths['processed_items'], index=False)
        print("Processed item data saved.")

        if provider == 'api':
            item_embeddings_np = get_embeddings_from_api(items_df['combined_text'].tolist(), model_name, dimensions)
        else:
            item_embeddings_np = get_embeddings_from_local(items_df['combined_text'].tolist(), model_name)

        np.save(item_emb_path, item_embeddings_np)
        print("Item embeddings generated and saved.")
    else:
        print("Cached item embeddings and processed file found, skipping generation.")

    # embed queries
    if not os.path.exists(query_emb_path):
        print("\nProcessing query data...")
        queries_df = pd.read_csv(paths['queries_data'])

        if provider == 'api':
            query_embeddings_np = get_embeddings_from_api(queries_df['search_term_pt'].tolist(), model_name, dimensions)
        else:
            query_embeddings_np = get_embeddings_from_local(queries_df['search_term_pt'].tolist(), model_name)

        np.save(query_emb_path, query_embeddings_np)
        print("Query embeddings generated and saved.")
    else:
        print("Cached query embeddings found, skipping generation.")

    print("\n--- data-process complete ---")