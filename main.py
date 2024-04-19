from dotenv import load_dotenv
import os, time
import pandas as pd
import streamlit as st
from IPython.display import Markdown
from llama_index.experimental import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini
# from llama_index.llms.openai import OpenAI
from prompts import instruction_str, new_prompt, context
from note_engine import note_engine
from pdf import canada_engine
from langchain_community.llms import Ollama

# Load environment variables (i.e. API Keys). Define the LLM (using google gemini)
load_dotenv()

# Join the data directory with the population csv file to define path
population_path = os.path.join("data", "population.csv")

# Read in data using pandas
population_df = pd.read_csv(population_path)

# Create a query engine: convert natural language to Pandas python code using LLMs. Pass the templates to engine
population_query_engine = PandasQueryEngine(df=population_df, verbose=False, instruction_str=instruction_str)
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
llm_mistral = Ollama(model="mistral")
agent = ReActAgent.from_tools(tools=tools, llm=llm_gemini, verbose=True, context=context)

##initialize our streamlit app
st.set_page_config(page_title="AI RAG Demo")
st.header("Gemini Application")
input = st.text_input("Input: ", key="input")
submit = st.button("Ask a question about population or Canada")
exit_app = st.button("Close")

if exit_app:
    time.sleep(5)
    st.query_params(close=True)
    st.markdown("Closing...")
    st.stop()

if submit:
    result = agent.query(input)
    st.subheader("The Response is")
    st.write(result.response)