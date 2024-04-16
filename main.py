from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini
# from llama_index.llms.openai import OpenAI
from prompts import instruction_str, new_prompt, context
from note_engine import note_engine
from pdf import canada_engine

# Load environment variables (i.e. API Keys). Define the LLM (using google gemini)
load_dotenv()

# Join the data directory with the population csv file to define path
population_path = os.path.join("data", "population.csv")

# Read in data using pandas
population_df = pd.read_csv(population_path)

# Create a query engine: convert natural language to Pandas python code using LLMs. Pass the templates to engine
population_query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompts": new_prompt})

# Tools are defined with the documentation that the LLM can reference (i.e. RAG)
tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, metadata=ToolMetadata(
        name="population_data",
        description="This gives information of the world population and demographics",
    )),
    QueryEngineTool(query_engine=canada_engine, metadata=ToolMetadata(
        name="canada_data",
        description="This gives detailed information about Canada the country",
    )),
]

llm_gemini = Gemini(model="gemini-pro",google_api_key=os.getenv("GOOGLE_API_KEY"))
# llm_openai = OpenAI(model="gpt-3.5-turbo-0613", openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = ReActAgent.from_tools(tools=tools, llm=llm_gemini, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)


