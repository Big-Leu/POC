import streamlit as st

st.set_page_config(
    page_title="Personal POC Repository",
    page_icon="🧪",
)

st.write("# Welcome to the Personal POC Repository! 🧪")

st.sidebar.success("Select a module from the sidebar.")

st.markdown(
    """
    This repository contains various Proof of Concepts (POCs) for personal
    learning and experimentation.

    ## Available Modules
    - **Duplicate Detection App**: Upload a CSV and find potential duplicates
      using exact, fuzzy, and semantic matching.
    - **Control Recommendation System**: Match controls to a reference dataset of
      recommended controls using multiple matching techniques.

    ## How to Use
    - Use the sidebar to navigate to the available modules.
    - Each module includes instructions and interactive features.
    - Upload a CSV file in each module to perform analysis.
    - Adjust thresholds and parameters to fine-tune matching results.

    ## Features
    - **Exact Matching**: Hash-based optimization for fast and precise matching
    - **Fuzzy Matching**: Levenshtein distance-based text similarity
    - **Semantic Matching**: NLP embeddings for meaning-based similarity
    
    ## Resources
    - See the [README](../README.md) for setup and details.
    - Explore the code in the `backend/` folder for implementation.
    """
)
