"""
Utility functions for working with language models and embedding models.

This module provides functions for loading and using embedding models 
through Langchain integrations.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

class HuggingFaceEmbeddings:
    """
    A wrapper class for using HuggingFace SentenceTransformers with a consistent API.
    
    This class allows us to easily swap between embedding providers while maintaining
    the same interface.
    """
    
    def __init__(self, model_name="all-mpnet-base-v2"):
        """
        Initialize the HuggingFace embeddings model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts, batch_size=32, show_progress=True):
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show a progress bar
            
        Returns:
            List of embeddings, one for each text
        """
        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress
        ).tolist()
    
    def embed_query(self, text):
        """
        Generate an embedding for a single query text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding for the text
        """
        return self.model.encode(text).tolist()

@st.cache_resource
def get_embeddings_model(provider="huggingface", model_name="all-mpnet-base-v2"):
    """
    Load an embeddings model with caching.
    
    Args:
        provider: The embedding provider to use ('huggingface' or 'openai')
        model_name: The name of the model to use
        
    Returns:
        An embeddings model object
    """
    if provider.lower() == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        # Default fallback to HuggingFace
        return HuggingFaceEmbeddings(model_name=model_name)
