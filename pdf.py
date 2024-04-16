import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.llms.gemini import Gemini
load_dotenv()

llm_gemini = Gemini(model="gemini-pro",google_api_key=os.getenv("GOOGLE_API_KEY"))
# llm_openai = OpenAI(model="gpt-3.5-turbo-0613", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define a function to get vector store index
def get_index(data, index_name):
    # Initialize index
    index = None

    # Does the index exist? If not, build the index. If it does exist, load from storage
    if not os.path.exists(index_name):
        print("Building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))

    return index

# Join the data directory with the Canada PDF file to define path
pdf_path = os.path.join("data", "Canada.pdf")

# Read in data using pandas
canada_pdf = PDFReader().load_data(file=pdf_path)

canada_index = get_index(canada_pdf,"canada")
canada_engine = canada_index.as_query_engine(llm=llm_gemini)
