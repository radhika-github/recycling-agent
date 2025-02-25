import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.core import Settings

# Function to set embedding model and disable OpenAI LLM
def configure_settings(embed_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Configure embedding model and LLM settings."""
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    Settings.llm = MockLLM()

# Function to create or load an index
def get_index(data, index_name):
    """Create or load an index with local embeddings."""
    if not os.path.exists(index_name):
        print(f"Creating new index: {index_name}")
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))

    return index

# Function to process a PDF and return a query engine
def process_pdf(pdf_path, index_name):
    """Process a PDF file and return a query engine."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    pdf_data = PDFReader().load_data(file=pdf_path)
    index = get_index(pdf_data, index_name)
    return index.as_query_engine()

# Configure settings with default embedding model
configure_settings()

# Example usage
pdf_file = os.path.join("data", "sample.pdf")
index_name = "sample_index"

query_engine = process_pdf(pdf_file, index_name)
