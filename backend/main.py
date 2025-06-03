from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Switch from OpenAI embeddings to HuggingFace embeddings (free alternative)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
import os
from langchain_community.vectorstores import FAISS

# You can keep this line but it won't be used with HuggingFace embeddings


loader = TextLoader("D:/GitHub/Personal/POC/backend/test.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
docs = text_splitter.split_documents(documents)

# Replace OpenAI embeddings with HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create FAISS index
library = FAISS.from_documents(docs, embeddings)
Query = "Who replaced CLiff Burton in Metallica?"
Query_Answer = library.similarity_search(Query)
docs_and_scores = library.similarity_search_with_score(Query)
print(docs_and_scores[0])
retriever = library.as_retriever()

# Save the index
library.save_local("faiss_index")

# Load the index with allow_dangerous_deserialization=True
# This is safe since you created this index yourself
metallica_saved = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # Add this parameter
)

# Replace OpenAI with a Hugging Face model to avoid API limits
from langchain_huggingface import HuggingFaceEndpoint

# Try to use a free model endpoint if OpenAI still fails
try:
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=metallica_saved.as_retriever(),
    )
    results = qa.invoke(Query)
    print(results["result"])
except Exception as e:
    print(f"Error using OpenAI: {e}")
    print("Falling back to basic retrieval...")
    # Just print the top results from the retriever
    for doc in Query_Answer:
        print(doc.page_content)