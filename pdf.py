import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("Creating new index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))
        
    return index

# Use a local Hugging Face embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# Load the LLM properly
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set local Hugging Face LLM
local_llm = HuggingFaceLLM(model=hf_model, tokenizer=hf_tokenizer)
Settings.llm = local_llm

pdf_path = os.path.join("data", "India.pdf")
India_pdf = PDFReader().load_data(file=pdf_path)
India_index = get_index(India_pdf, "India_index")
India_engine = India_index.as_query_engine()