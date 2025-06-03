"""
Shared utility functions for the POC application.

This module contains shared functionality used across different modules
of the application, including data loading, matching algorithms, and UI helpers.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----- Model and Embedding Functions -----


@st.cache_resource
def load_model(model_name="all-mpnet-base-v2"):
    """Load the sentence transformer model with caching.

    Args:
        model_name: The name of the SentenceTransformer model to load

    Returns:
        A SentenceTransformer model instance
    """
    return SentenceTransformer(model_name)


def generate_embeddings(texts, model=None, show_progress=True, batch_size=32):
    """Generate embeddings for the provided texts.

    Args:
        texts: List of text strings to embed
        model: SentenceTransformer model (if None, will load the default model)
        show_progress: Whether to show a progress bar
        batch_size: Batch size for embedding generation

    Returns:
        numpy.ndarray of text embeddings
    """
    if model is None:
        model = load_model()
    return model.encode(
        texts, show_progress_bar=show_progress, batch_size=batch_size
    )


def save_embeddings(embeddings, file_path):
    """Save embeddings to a numpy file.

    Args:
        embeddings: numpy.ndarray of embeddings to save
        file_path: Path where to save the embeddings

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Make sure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, embeddings)
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False


def load_embeddings(file_path):
    """Load embeddings from a numpy file.

    Args:
        file_path: Path to the numpy embeddings file

    Returns:
        numpy.ndarray of embeddings or None if file doesn't exist or error occurs
    """
    try:
        if os.path.exists(file_path):
            return np.load(file_path)
        return None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None


# ----- Data Loading and Preparation Functions -----


def load_csv_with_error_handling(uploaded_file, sample_size=None):
    """Load a CSV file with robust error handling.

    Args:
        uploaded_file: The uploaded file object
        sample_size: If set, only load this many rows (for testing)

    Returns:
        pandas.DataFrame: The loaded dataframe
    """
    try:
        # First attempt - standard settings
        if sample_size:
            df = pd.read_csv(uploaded_file, nrows=sample_size)
        else:
            df = pd.read_csv(uploaded_file)
    except pd.errors.ParserError:
        uploaded_file.seek(0)  # Reset file pointer
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines="skip")
            st.warning("Some rows were skipped due to formatting issues.")
        except Exception as e:
            # If all else fails, try with the Python engine
            uploaded_file.seek(0)
            if sample_size:
                df = pd.read_csv(
                    uploaded_file,
                    engine="python",
                    on_bad_lines="skip",
                    nrows=sample_size,
                )
            else:
                df = pd.read_csv(
                    uploaded_file,
                    engine="python",
                    on_bad_lines="skip",
                )
            st.warning(f"Loaded CSV with some issues: {str(e)}")

    # Handle empty dataframes
    if df.empty:
        st.error("The uploaded file is empty or couldn't be parsed correctly.")
        return None

    return df


def prepare_combined_text(df, columns):
    """Create a combined text field from multiple columns.

    Joins the specified columns with spaces to create a single text field
    for similarity calculations.

    Args:
        df: The dataframe containing the columns
        columns: List of column names to combine

    Returns:
        pandas.Series: Combined text column
    """
    return df[columns].astype(str).agg(" ".join, axis=1)


def save_dataframe(df, directory, filename, formats=None):
    """Save a dataframe to disk in multiple formats.

    Args:
        df: The dataframe to save
        directory: The directory to save the file to
        filename: The base filename (without extension)
        formats: List of formats to save (csv, json supported)
    """
    if formats is None:
        formats = ["csv", "json"]

    os.makedirs(directory, exist_ok=True)

    for fmt in formats:
        filepath = os.path.join(directory, f"{filename}.{fmt}")
        if fmt == "csv":
            df.to_csv(filepath, index=False)
        elif fmt == "json":
            data = df.to_dict(orient="records")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)


# ----- Matching Functions -----


def create_hash_map(df, columns):
    """Create a hash map for fast exact matching.

    Args:
        df: The dataframe containing the records
        columns: List of column names to use for hashing

    Returns:
        dict: Mapping from hash values to row indices
    """
    hash_map = {}
    for idx, row in df[columns].astype(str).iterrows():
        row_tuple = tuple(row)
        row_hash = hash(row_tuple)
        if row_hash not in hash_map:
            hash_map[row_hash] = []
        hash_map[row_hash].append(idx)
    return hash_map


def find_exact_matches(row_data, compare_data, columns):
    """Check if two records match exactly on the specified columns.

    Args:
        row_data: First record as a pandas Series
        compare_data: Second record as a pandas Series
        columns: List of column names to compare

    Returns:
        bool: True if all specified columns match exactly
    """
    return all(row_data[col] == compare_data[col] for col in columns)


def calculate_fuzzy_score(row_data, compare_data, columns, weights=None):
    """Calculate fuzzy match score between two records.

    Uses the Levenshtein distance to compute string similarity between records.

    Args:
        row_data: First record as a pandas Series
        compare_data: Second record as a pandas Series
        columns: List of column names to compare
        weights: Optional dict of column: weight pairs (defaults to equal weights)

    Returns:
        float: The weighted average fuzzy match score (0-100)
    """
    fuzzy_scores = []
    total_weight = 0

    # If no weights provided, use equal weights
    if weights is None:
        weights = {col: 1 for col in columns}

    for col in columns:
        weight = weights.get(col, 1)
        if pd.notna(row_data[col]) and pd.notna(compare_data[col]):
            score = fuzz.ratio(str(row_data[col]), str(compare_data[col]))
            fuzzy_scores.append(score * weight)
            total_weight += weight

    # Return average score or 0 if no valid comparisons
    return sum(fuzzy_scores) / total_weight if total_weight > 0 else 0


def calculate_semantic_similarity(embeddings1, embeddings2):
    """
    Calculate cosine similarity between two embedding vectors

    Args:
        embeddings1: First embedding vector or array
        embeddings2: Second embedding vector or array

    Returns:
        float or numpy.ndarray: Cosine similarity score(s) (0-1)
    """
    # Ensure we have 2D arrays
    if len(embeddings1.shape) == 1:
        embeddings1 = embeddings1.reshape(1, -1)
    if len(embeddings2.shape) == 1:
        embeddings2 = embeddings2.reshape(1, -1)

    return cosine_similarity(embeddings1, embeddings2)


def sort_matches_by_priority(matches):
    """
    Sort matches by type (Exact > Fuzzy > Semantic) and score

    Args:
        matches: List of match dictionaries with match_type and score

    Returns:
        list: Sorted matches
    """

    def sort_key(x):
        type_priority = {"Exact": 0, "Fuzzy": 1, "Semantic": 2}
        return (type_priority.get(x["match_type"], 3), -x["score"])

    return sorted(matches, key=sort_key)


def find_top_matches(
    query_embeddings, reference_embeddings, threshold=0.75, top_n=5
):
    """
    Find top semantic matches between query and reference embeddings

    Args:
        query_embeddings: Embeddings of query records
        reference_embeddings: Embeddings of reference records
        threshold: Minimum similarity threshold (0-1)
        top_n: Number of top matches to return

    Returns:
        list: Lists of indices and scores for top matches
    """
    # Calculate all similarities
    similarity_matrix = cosine_similarity(
        query_embeddings, reference_embeddings
    )

    top_indices = []
    top_scores = []

    # Find top matches for each query
    for i in range(similarity_matrix.shape[0]):
        # Get scores for this query
        scores = similarity_matrix[i]

        # Sort scores and get indices of top matches above threshold
        indices = np.argsort(scores)[::-1]
        filtered_indices = [
            idx for idx in indices if scores[idx] >= threshold
        ][:top_n]

        # Get corresponding scores
        filtered_scores = [scores[idx] for idx in filtered_indices]

        top_indices.append(filtered_indices)
        top_scores.append(filtered_scores)

    return top_indices, top_scores


# ----- Caching Functions -----


@st.cache_data
def cache_similarity_matrix(uploaded_embeddings, recommended_embeddings):
    """
    Cache the similarity matrix calculation between two embedding sets

    Args:
        uploaded_embeddings: Embeddings of uploaded controls
        recommended_embeddings: Embeddings of recommended controls

    Returns:
        numpy.ndarray: The similarity matrix
    """
    similarity_matrix = np.zeros((len(uploaded_embeddings), len(recommended_embeddings)))

    for i in range(len(uploaded_embeddings)):
        for j in range(len(recommended_embeddings)):
            similarity_matrix[i, j] = np.dot(
                uploaded_embeddings[i], recommended_embeddings[j]
            ) / (
                np.linalg.norm(uploaded_embeddings[i]) *
                np.linalg.norm(recommended_embeddings[j])
            )

    return similarity_matrix


@st.cache_data
def cache_fuzzy_scores(df1, df2, columns):
    """
    Cache fuzzy scores between two dataframes for faster lookups

    Args:
        df1: First dataframe (usually user uploaded controls)
        df2: Second dataframe (usually recommended controls)
        columns: List of columns to compare

    Returns:
        dict: Mapping of (i, j) to fuzzy score
    """
    scores = {}
    for i in range(len(df1)):
        row_data = df1.iloc[i]
        for j in range(len(df2)):
            rec_control = df2.iloc[j]
            score = calculate_fuzzy_score(row_data, rec_control, columns)
            scores[(i, j)] = score

    return scores


def export_matches_to_csv(results, uploaded_df, recommended_df,
                     user_columns, rec_columns, output_path=None):
    """Export matching results to CSV.

    Args:
        results: List of match results
        uploaded_df: User's uploaded controls dataframe
        recommended_df: Recommended controls dataframe
        user_columns: Columns from user data to include
        rec_columns: Columns from recommended data to include
        output_path: Path to save the CSV file (if None, returns the dataframe)

    Returns:
        pandas.DataFrame or None: If output_path is None, returns the dataframe
    """
    rows = []

    for result in results:
        original_idx = result["original_idx"]
        original_data = {}

        # Add original data with prefixes
        for col in user_columns:
            if col in uploaded_df.columns:
                original_data[f"uploaded_{col}"] = uploaded_df.iloc[original_idx][col]

        # Add matches
        for i, match in enumerate(result["matches"]):
            # For each match, create a new row with original data plus match data
            match_row = original_data.copy()

            # Add match metadata
            match_row["match_number"] = i + 1
            match_row["match_type"] = match["match_type"]
            match_row["match_score"] = match["score"]

            # Add matched data with prefixes
            for col in rec_columns:
                if col in recommended_df.columns:
                    match_row[f"recommended_{col}"] = match["data"][col]

            rows.append(match_row)

    # Create dataframe from all rows
    export_df = pd.DataFrame(rows)

    # Save to CSV if path provided
    if output_path:
        export_df.to_csv(output_path, index=False)
        return None

    return export_df


# ----- UI and Progress Tracking Functions -----


def setup_progress_tracking():
    """Create and return progress bar and status text elements"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text


def update_progress(progress_bar, status_text, stage, progress, message=None):
    """Update progress bar and status text"""
    progress_bar.progress(progress)
    if message:
        status_text.write(message)
    else:
        status_text.write(f"Stage: {stage}")


def format_match_result(match_type, score):
    """
    Return color-coded HTML for match type and score

    Args:
        match_type: Type of match (Exact, Fuzzy, Semantic)
        score: Match score

    Returns:
        str: HTML formatted string with color coding
    """
    match_color = {"Exact": "red", "Fuzzy": "orange", "Semantic": "green"}.get(
        match_type, "blue"
    )

    return f"<span style='color:{match_color}'>{match_type} match ({score:.2f}%)</span>"


def get_recommended_column_selection(df_columns, reference_columns):
    """
    Suggest columns to select based on reference column names

    Args:
        df_columns: List of column names in the uploaded dataframe
        reference_columns: List of reference column names to match against

    Returns:
        list: Recommended column selections
    """
    recommended = []

    # Direct matches
    for col in df_columns:
        if col.lower() in [ref.lower() for ref in reference_columns]:
            recommended.append(col)

    # If no direct matches, look for partial matches
    if not recommended:
        for ref in reference_columns:
            for col in df_columns:
                if ref.lower() in col.lower():
                    recommended.append(col)
                    if len(recommended) >= 5:  # Limit to 5 recommendations
                        break

    # If still no matches, return first few columns
    if not recommended and df_columns:
        recommended = df_columns[: min(5, len(df_columns))]

    return recommended


def timer_decorator(func):
    """
    Decorator to measure and display execution time

    Args:
        func: The function to time

    Returns:
        wrapper: The wrapped function
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        st.info(f"Execution time: {end_time - start_time:.2f} seconds")
        return result

    return wrapper
