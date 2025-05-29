import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Add parent directory to path for module imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now we can import from utils
from utils import (
    calculate_fuzzy_score, calculate_semantic_similarity,
    find_top_matches, format_match_result, generate_embeddings,
    get_recommended_column_selection, load_csv_with_error_handling,
    load_model, prepare_combined_text, save_dataframe,
    setup_progress_tracking, sort_matches_by_priority,
    timer_decorator, update_progress
)

st.title("Control Recommendation System")
st.markdown(
    """
This app helps you match your controls against a reference dataset of recommended controls.
Upload your controls as a CSV file, select columns for matching, and the app will:

1. Find exact matches for your controls
2. Find fuzzy matches based on text similarity 
3. Find semantic matches based on meaning similarity

You can adjust the thresholds to fine-tune the matching results.
"""
)


@st.cache_data
def load_recommended_controls():
    """Load and prepare the recommended controls dataset"""
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "recommended_controls.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    recommended_controls = [
        {
            "objective": "Authority and Purpose",
            "name": "Authority and Purpose",
            "class": "Preventive",
            "type": "Roles and Responsibilities",
            "description": (
                "This section establishes the authority of the Bureau of Consumer Financial Protection to issue Regulation B."
            ),
        },
        {
            "objective": "Authority and Purpose",
            "name": "Definitions",
            "class": "Preventive",
            "type": "Communicate",
            "description": (
                "This section defines key terms used in the regulation to ensure consistent interpretation across entities."
            ),
        },
        {
            "objective": "Authority and Purpose",
            "name": "General Rules",
            "class": "Preventive",
            "type": "Business Processes",
            "description": (
                "This section outlines general rules prohibiting discrimination in credit transactions."
            ),
        },
        {
            "objective": "General Rules",
            "name": "Record Retention",
            "class": "Detective",
            "type": "Data Management",
            "description": (
                "This section outlines requirements for retaining credit application records and documentation."
            ),
        },
        {
            "objective": "General Rules",
            "name": "Enforcement",
            "class": "Corrective",
            "type": "Audits and Risk Management",
            "description": (
                "This section describes enforcement mechanisms and penalties for Equal Credit Opportunity Act violations."
            ),
        },
    ]

    df_controls = pd.DataFrame(recommended_controls)
    df_controls.to_csv(csv_path, index=False)
    with open(os.path.join(data_dir, "recommended_controls.json"), "w") as f:
        json.dump(recommended_controls, f, indent=2)

    return df_controls


recommended_controls_df = load_recommended_controls()
controls_columns = ["objective", "name", "class", "type", "description"]
recommended_controls_df["combined_text"] = prepare_combined_text(
    recommended_controls_df, controls_columns
)


@st.cache_resource
def get_model_and_embeddings(texts):
    model = load_model()
    embeddings = generate_embeddings(texts, model, show_progress=True)
    return model, embeddings


model, recommended_embeddings = get_model_and_embeddings(
    recommended_controls_df["combined_text"].tolist()
)

st.info(f"Loaded {len(recommended_controls_df)} recommended controls")

with st.expander("View Recommended Controls Dataset"):
    st.dataframe(recommended_controls_df[controls_columns])

st.subheader("Upload Your Controls for Analysis")
uploaded_file = st.file_uploader(
    "Upload a CSV or JSON file containing controls", type=["csv", "json"]
)

df = None
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "json":
        try:
            data = json.load(uploaded_file)
            df = pd.DataFrame(data) if isinstance(data, (list, dict)) else None
            if df is not None and not df.empty:
                st.success(f"Loaded JSON with {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")
    elif file_type == "csv":
        df = load_csv_with_error_handling(uploaded_file)
        if df is not None and not df.empty:
            st.success(f"Loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")

if df is not None and not df.empty:
    with st.expander("Data Preview"):
        st.dataframe(df.head())

    st.subheader("Select Columns for Matching")
    available_columns = df.columns.tolist()
    default_columns = get_recommended_column_selection(available_columns, controls_columns)

    selected_columns = st.multiselect(
        "Select columns to use for matching controls",
        available_columns,
        default=default_columns,
    )

    col1, col2 = st.columns(2)
    with col1:
        fuzzy_threshold = st.slider("Fuzzy Match Threshold (%)", 50, 100, 80)
    with col2:
        semantic_threshold = st.slider("Semantic Similarity Threshold (%)", 50, 100, 75)

    num_recommendations = st.slider(
        "Number of recommendations to show per control", 1, 10, 3
    )

    if st.button("Find Matching Controls"):
        if not selected_columns:
            st.error("Please select at least one column for analysis")
        else:
            df["combined_text"] = prepare_combined_text(df, selected_columns)
            progress_bar, status_text = setup_progress_tracking()

            update_progress(progress_bar, status_text, "Embeddings", 0, "Generating embeddings for uploaded controls...")
            uploaded_embeddings = generate_embeddings(df["combined_text"].tolist(), model)
            update_progress(progress_bar, status_text, "Embeddings", 40)

            similarity_matrix = np.zeros((len(df), len(recommended_controls_df)))
            for i in range(len(df)):
                for j in range(len(recommended_controls_df)):
                    similarity_matrix[i, j] = np.dot(uploaded_embeddings[i], recommended_embeddings[j]) / (
                        np.linalg.norm(uploaded_embeddings[i]) * np.linalg.norm(recommended_embeddings[j])
                    )

            update_progress(progress_bar, status_text, "Similarity", 70)

            results = []
            for i in range(len(df)):
                row_data = df.iloc[i]
                potential_matches = []

                for j in range(len(recommended_controls_df)):
                    rec_control = recommended_controls_df.iloc[j]
                    
                    # Count columns that exist in both dataframes
                    comparable_columns = [col for col in selected_columns 
                                        if col in recommended_controls_df.columns]
                    
                    total_comparisons = len(comparable_columns)
                    
                    # Only consider as exact match if ALL columns match exactly (100%)
                    if total_comparisons > 0:
                        exact_match = True
                        for col in comparable_columns:
                            if str(row_data[col]).lower() != str(rec_control[col]).lower():
                                exact_match = False
                                break
                                
                        if exact_match:
                            potential_matches.append({
                                "index": j, 
                                "match_type": "Exact", 
                                "score": 100.0
                            })

                if len(potential_matches) < num_recommendations:
                    for j in range(len(recommended_controls_df)):
                        if any(match["index"] == j for match in potential_matches):
                            continue
                        rec_control = recommended_controls_df.iloc[j]
                        avg_fuzzy_score = calculate_fuzzy_score(row_data, rec_control, selected_columns)
                        if avg_fuzzy_score >= fuzzy_threshold:
                            potential_matches.append({"index": j, "match_type": "Fuzzy", "score": avg_fuzzy_score})

                if len(potential_matches) < num_recommendations:
                    for j in range(len(recommended_controls_df)):
                        if any(match["index"] == j for match in potential_matches):
                            continue
                        semantic_score = similarity_matrix[i, j] * 100
                        if semantic_score >= semantic_threshold:
                            potential_matches.append({"index": j, "match_type": "Semantic", "score": semantic_score})

                potential_matches = sort_matches_by_priority(potential_matches)
                top_matches = potential_matches[:num_recommendations]

                if top_matches:
                    original_data = {col: row_data[col] for col in selected_columns}
                    matches = []
                    for match in top_matches:
                        match_data = {col: recommended_controls_df.iloc[match["index"]][col] for col in controls_columns}
                        matches.append({"idx": match["index"], "data": match_data, "match_type": match["match_type"], "score": match["score"]})

                    results.append({"original_idx": i, "original_data": original_data, "matches": matches})

            update_progress(progress_bar, status_text, "Complete", 100, "Analysis complete!")

            st.subheader("Control Matching Results")

            if not results:
                st.info("No matches found with the current settings.")
            else:
                st.write(f"Found matches for {len(results)} controls.")
                for result in results:
                    with st.expander(f"Control #{result['original_idx'] + 1}"):
                        st.write("**Your Control:**")
                        for col, val in result["original_data"].items():
                            st.write(f"- **{col}:** {val}")

                        st.write("**Recommended Controls:**")
                        for j, match in enumerate(result["matches"]):
                            formatted_match = format_match_result(match["match_type"], match["score"])
                            st.markdown(f"**Match {j+1}:** {formatted_match}", unsafe_allow_html=True)
                            with st.container():
                                cols = st.columns([1, 3])
                                with cols[0]:
                                    st.write("**Objective:**")
                                    st.write("**Name:**")
                                    st.write("**Class:**")
                                    st.write("**Type:**")
                                with cols[1]:
                                    st.write(match["data"]["objective"])
                                    st.write(match["data"]["name"])
                                    st.write(match["data"]["class"])
                                    st.write(match["data"]["type"])
                                st.write("**Description:**")
                                st.write(match["data"]["description"])
                                st.divider()
else:
    st.info("Please upload a CSV or JSON file with controls to find matches.")