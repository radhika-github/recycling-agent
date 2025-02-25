import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings  # ✅ Override default settings
from llama_index.llms.llama_cpp import LlamaCPP  # ✅ Correct wrapper for Llama-cpp

# Load environment variables
load_dotenv()

# ✅ Use a free, open-source embedding model (runs locally)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model

# ✅ Load Mistral Locally with Llama-CPP wrapper
llm = LlamaCPP(
    model_path="mistral-7b-v0.1.Q4_K_M.gguf",  # Ensure the path is correct
    temperature=0.7,
    max_new_tokens=512,
    context_window=4096,
    verbose=True
)

# ✅ Test if the LLM is working
response = llm.complete("What is the capital of France?")
print(response.text)  # Expected output: "Paris"

# ✅ Assign to LlamaIndex settings
Settings.llm = llm  

# ✅ Load dataset
# population_path = os.path.join("data", "WorldPopulation2023.csv")
# population_df = pd.read_csv(population_path)

# # ✅ Set up query engine for population data
# population_query_engine = PandasQueryEngine(df=population_df, verbose=True)
# population_query_engine.update_prompts({"pandas_prompt": "Answer using the population dataset."})

# ✅ Define tools
# tools = [
#     QueryEngineTool(query_engine=population_query_engine, metadata=ToolMetadata(
#         name="population_data",
#         description="This gives information about world population and demographics"),
#     ),
# ]

tools = []
# ✅ Create an agent using Llama-CPP
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context="context")

# ✅ User input loop
while (prompt := input("Enter a prompt (q to Quit): ")) != "q":
    result = agent
