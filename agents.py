from tools import llm, evaluate_context, deepsearch, openai_api_key
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.agent.workflow import FunctionAgent
from ingest_anything.agent import IngestAgent
import os

os.environ["OPENAI_API_KEY"] = openai_api_key

web_searching_agent = FunctionAgent(
    name = "WebSearchingAgent",
    description="Useful for searching the web about books of all kinds!",
    system_prompt="""You are a web-searching assistant for a bibliophile: your task is to research the depths of web, using the 'deepsearch' tool: this tool will return you a JSON object with the information about the books - your task is to summarize that information for the user, evaluate its correctness and relevancy with the 'evaluate_context' tool, and then, once you have a relevant and correct answer, return it to the user. Please dismiss any query which is not about books or literature in general.""",
    tools=[deepsearch, evaluate_context]
)

qdrant_client = QdrantClient("http://qdrant:6333/")
async_qdrant_client = AsyncQdrantClient("http://qdrant:6333/")
vector_store = QdrantVectorStore(client=qdrant_client, aclient=async_qdrant_client, collection_name="library")

agent_factory = IngestAgent()
ingest_agent = agent_factory.create_agent(
    vector_database=vector_store,
    llm=llm,
    ingestion_type="anything", 
    agent_type="react", 
    tools = [evaluate_context],
    query_transform="hyde"
)
library_agent = ingest_agent.get_agent(name="LibraryAgent", description="Useful for searching a library vector database for information about information on books contained in it",system_prompt="You are LibraryAgent, and you're in charge of retrieving information from a library database of books, based on the user query: using your 'query_engine_tool', you will be able to get the information you need from the library vector database, information that you will then need to evaluate with the 'evaluate_context' tool. If you cannot find reliable and relevant information, please tell the user that you cannot give them an answer.")
