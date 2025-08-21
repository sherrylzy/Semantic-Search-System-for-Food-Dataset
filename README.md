# Semantic Search System for Food Dataset

This project implements a semantic search system for a food dataset as part of the Prosus AI Internship Technical Assignment.

It processes natural language queries in Portuguese and retrieves the most relevant food items from a database of 5,000 items.

## Project Structure

```
.
├── data/                     # Contains the provided CSV files
├── embeddings/               # Stores generated vector embeddings
├── config.yaml               # All project configurations
├── utils.py                  # Utility functions
├── step1_generate_embeddings.py  # Script to generate embeddings
├── step2_build_search_engine.py  # The core search engine logic
├── step3_evaluate_results.py     # LLM-based evaluation logic
├── main.py                   # Main script to run the full pipeline
└── README.md                 # This file
```

## Setup

1.  **Clone the repository** (or create the files as described).

2.  **Create a Conda environment** and install dependencies:
    ```bash
    conda create -n foodsearch python=3.9
    conda activate foodsearch
    pip install pandas numpy tqdm litellm scikit-learn faiss-cpu pyyaml
    ```

3.  **Add Data**: Place `5k_items_curated.csv` and `queries.csv` into the `data/` directory.

4.  **Configure API Key**: Open `config.yaml` and ensure your `api.key` is set correctly.

## How to Run

To run the complete pipeline (generate embeddings, search, and evaluate), simply execute the `main.py` script from the project root directory:

```bash
python main.py
```

The script will:
1.  Read the data and generate embeddings for all food items and queries (this will only happen on the first run).
2.  Build a `faiss` index for efficient searching.
3.  Execute a search for all 100 queries and save the top 5 results to `search_results.json`.
4.  Perform an LLM-based evaluation on the top result for a small sample of queries and print the average relevance score.
