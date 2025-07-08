# Personal POC Repository

This repository contains various Proof of Concepts (POCs) for personal learning and experimentation.

## Contents

- **Duplicate Detection Module**: Find potential duplicates in CSV data
- **Control Recommendation Module**: Match controls against a reference dataset
- Shared utilities for text processing and matching
- Interactive Streamlit UI

---

## Modules

### 1. Duplicate Detection App

This project includes a powerful **Duplicate Detection App** built with Streamlit and Python. It allows you to upload a CSV file and find potential duplicate records using a hybrid approach:

- **Exact Match**: Identifies records that are identical across selected columns.
- **Fuzzy Match**: Uses text similarity (Levenshtein distance) to find near-duplicates (threshold adjustable).
- **Semantic Similarity**: Uses sentence embeddings (via `sentence-transformers`) to find records with similar meaning (threshold adjustable).

### Features

- **Robust CSV Parsing**: Handles malformed or inconsistent CSV files gracefully.
- **Column Selection**: Choose which columns to use for duplicate detection.
- **Threshold Controls**: Adjust fuzzy and semantic similarity thresholds interactively.
- **Progress Bar**: Visual feedback during processing, suitable for large datasets.
- **Prioritized Results**: Shows top 3 duplicates per record, prioritizing Exact > Fuzzy > Semantic matches.
- **Expandable Results**: View details for each record and its top duplicates.

---

## How It Works

1. **Upload CSV**: The app accepts a CSV file with any number of columns. You can select up to 4 columns for analysis (default: first 4).
2. **Data Preview**: See a preview of your data before running analysis.
3. **Set Thresholds**: Use sliders to set the minimum similarity for fuzzy and semantic matches (default: 90%).
4. **Run Detection**: Click 'Detect Duplicates' to start. The app will:
    - Combine selected columns into a single text field per row.
    - Generate sentence embeddings for semantic similarity.
    - Compute a similarity matrix for all records.
    - For each record, compare with all others:
        - If all selected columns match exactly, it's an **Exact** match.
        - If fuzzy text similarity (average across columns) exceeds the threshold, it's a **Fuzzy** match.
        - If semantic similarity exceeds the threshold, it's a **Semantic** match.
    - Only the highest-priority match type is kept for each pair.
    - Top 3 matches per record are shown, sorted by priority and score.
5. **View Results**: Expand each record to see its top potential duplicates, with color-coded match types.

---

## How to Run

### Prerequisites
- Python 3.8+
- [Poetry](https://python-poetry.org/) (recommended for dependency management)

### Install Dependencies

If using Poetry:
```bash
poetry install
```


### Start the App
```bash
poetry run streamlit run backend/main.py
```

Or without Poetry:
```bash
pip install -r requirements.txt
streamlit run backend/main.py
```

The app will open in your browser. Use the navigation menu to select modules.

---

### 2. Control Recommendation System

This module helps you match your controls against a reference dataset of recommended controls. It features:

- **Pre-loaded Reference Data**: Comes with sample controls that serve as a benchmark
- **Multiple Matching Approaches**: Uses exact, fuzzy, and semantic matching
- **Smart Column Selection**: Automatically suggests relevant columns for matching
- **Adjustable Thresholds**: Fine-tune the matching sensitivity
- **Interactive Results**: View detailed information about each match

### Shared Utility Functions

The application leverages a robust set of shared utility functions in `utils.py`:

- **Embedding Generation**: Optimized sentence embedding creation
- **Smart CSV Loading**: Handles various CSV formats and errors
- **Efficient Matching Algorithms**: Hash-based exact matching, weighted fuzzy matching
- **Data Processing Helpers**: Text preparation, combined fields, and prioritization
- **UI Components**: Progress tracking, formatted results, and timing metrics

---

## Example CSV Format

### For Duplicate Detection
```
name,description,category,notes
Widget A,Small blue widget,Tools,For home use
Widget B,Small blu widget,Tools,For home use
Widget C,Large red widget,Gadgets,Outdoor
```

### For Control Recommendations
```
objective,name,class,type,description
Authorization,Access Control,Preventive,Technical,Control access to systems
Data Protection,Encryption,Preventive,Technical,Encrypt sensitive data
```

---

## Project Structure

```
POC/
├── backend/
│   ├── app.py               # Original duplicate detection code
│   ├── main.py              # Landing page for the Streamlit app
│   ├── utils.py             # Shared utility functions
│   ├── __init__.py
│   └── pages/
│       ├── 1_👀​_Duplicate_Detection_Module.py     # Module 1: Duplicate Detection
│       └── 2_🤖_ml_recommendations.py             # Module 2: Control Recommendations
├── data/                    # Generated data files (created on first run)
├── poetry.lock
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

Prompt:
Create a Python script that reads a CSV file and performs duplicate record matching using a tiered scoring system. For each record, compare it against all other records in the CSV and compute:

Exact Match – Field-by-field exact string equality.

Fuzzy Match – Use a string similarity metric (like Levenshtein ratio or fuzz.token_sort_ratio from fuzzywuzzy) when exact match fails.

Semantic Match – Use sentence embeddings (e.g., with SentenceTransformer) to compute cosine similarity when fuzzy match is insufficient.

Logic:

Compare each record with all others (excluding itself).

Assign match scores:

Exact match = 1.0

Fuzzy match (if similarity > threshold, e.g. 90) = 0.7

Semantic match (cosine similarity > threshold, e.g. 0.85) = 0.4

For each record, calculate:

A total aggregate match score based on the matches found.

A match percentage: (aggregate match score) / (max possible score from comparing with all other records).

Prioritize exact > fuzzy > semantic (i.e., once a higher-priority match is found, skip lower ones for that field).