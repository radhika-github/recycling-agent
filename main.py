from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context 
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.settings import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pdf import India_engine
from transformers import AutoTokenizer

load_dotenv()  

population_path = os.path.join("data", "WorldPopulation2023.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
# population_query_engine.query("What is the population on India")

tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, metadata=ToolMetadata(
        name="population_data",
        description="this gives information at the world population and demographics"),
                    ),
    
    QueryEngineTool(query_engine=India_engine, metadata=ToolMetadata(
        name="india_data",
        description="this gives detailed info about india"),
                    ),
]

# Set up free Hugging Face models
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{prompt}"  # Basic template; modify if needed

llm = HuggingFaceLLM(model_name="mistralai/Mistral-7B-Instruct-v0.1",
                     tokenizer=tokenizer)  
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Override default LlamaIndex settings to avoid OpenAI
Settings.embed_model = embed_model

# Ensure ReActAgent uses the correct models
agent = ReActAgent.from_tools(
    tools=tools, 
    llm=llm, 
    embed_model=embed_model,  # Explicitly set embedding model
    verbose=True, 
    context="context"
)

while(promt := input("Enter a prompt (q to Quit): ")) !="q":
    result = agent.query(promt)
    print(result)
# print(population_df.head())




