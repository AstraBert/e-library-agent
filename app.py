import os
import json
import requests as rq
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
import gradio as gr
from typing import List, Literal
from agents import ingest_agent, AgentWorkflow, web_workflow, openai_api_key
from llama_index.core.agent.workflow import ToolCall, ToolCallResult
import uvicorn
from utils import ChatHistory, ChatMessage

os.environ["OPENAI_API_KEY"] = openai_api_key

class AgentApiInput(BaseModel):
    prompt: str

class IngestApiInput(BaseModel):
    files: List[str]

class AgentApiOutput(BaseModel):
    response: str
    process: str

class IngestApiOutput(BaseModel):
    error_free: bool

app = FastAPI(default_response_class=ORJSONResponse)
HISTORY = ChatHistory()

@app.post("/ingest")
async def ingestion(inpt: IngestApiInput) -> IngestApiOutput:
    try:
        ingest_agent.ingest(files_or_dir=inpt.files, embedding_model="text-embedding-3-small", chunker="neural")
    except Exception:
        return IngestApiOutput(error_free=False)
    return IngestApiOutput(error_free=True)

@app.post("/search/library")
async def search(inpt: AgentApiInput) -> AgentApiOutput:
    library_agent = ingest_agent.get_agent(name="LibraryAgent", description="Useful for searching a library vector database for information about information on books contained in it",system_prompt="You are LibraryAgent, and you're in charge of retrieving information from a library database of books, based on the user query: using your 'query_engine_tool', you will be able to get the information you need from the library vector database, information that you will then need to evaluate with the 'evaluate_context' tool. If you cannot find reliable and relevant information, please tell the user that you cannot give them an answer.")
    library_workflow = AgentWorkflow(
        agents=[library_agent],
        root_agent=library_agent.name,
    )
    handler = library_workflow.run(user_msg=inpt.prompt, chat_history=HISTORY.messages)
    process = ""
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            process += f"Result for tool **{event.tool_name}**:\n\n```json\n{event.tool_output.model_dump_json(indent=4)}\n```\n\n"
        elif isinstance(event, ToolCall):
            process += f"Calling tool **{event.tool_name}** with input:\n\n```json\n{json.dumps(event.tool_kwargs, indent=4)}\n```\n\n"
    response = await handler
    response = str(response)
    HISTORY.messages.append(ChatMessage.from_str(content=inpt.prompt, role="user"))
    HISTORY.messages.append(ChatMessage.from_str(content=process, role="assistant"))
    HISTORY.messages.append(ChatMessage.from_str(content=response, role="assistant"))
    return AgentApiOutput(response=response, process=process)

@app.post("/search/web")
async def websearch(inpt: AgentApiInput) -> AgentApiOutput:
    handler = web_workflow.run(user_msg=inpt.prompt)
    process = ""
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            process += f"Result for tool **{event.tool_name}**:\n\n```json\n{event.tool_output.model_dump_json(indent=4)}\n```\n\n"
        elif isinstance(event, ToolCall):
            process += f"Calling tool **{event.tool_name}** with input:\n\n```json\n{json.dumps(event.tool_kwargs, indent=4)}\n```\n\n"
    response = await handler
    response = str(response)
    return AgentApiOutput(response=response, process=process)

def ingest_book(files: List[str]) -> Literal["Book ingestion was successful!", "There was an error during book ingestion :("]:
    res = rq.post(url="http://localhost:8000/ingest", json=IngestApiInput(files=files).model_dump())
    if res.json()["error_free"]:
        return "Book ingestion was successful!"
    else:
        return "There was an error during book ingestion :("

def search_library(message, history) -> str:
    res = rq.post(url="http://localhost:8000/search/library", json=AgentApiInput(prompt=message).model_dump())
    if res.status_code == 200:
        response = res.json()["response"]
        process = res.json()["process"]
        final_response = f"<details>\n\t<summary><b>Agentic Process</b></summary>\n\n{process}\n\n</details>\n\n{response}"
    else:
        final_response = f"There was an error in generating your response:\n\n<details>\n\t<summary><b>Error Logs</b></summary>\n\n{res.text}\n\n</details>\n\n"
    return final_response

def search_web(message, history) -> str:
    res = rq.post(url="http://localhost:8000/search/web", json=AgentApiInput(prompt=message).model_dump())
    if res.status_code == 200:
        response = res.json()["response"]
        process = res.json()["process"]
        final_response = f"<details>\n\t<summary><b>Agentic Process</b></summary>\n\n{process}\n\n</details>\n\n{response}"
    else:
        final_response = f"There was an error in generating your response:\n\n<details>\n\t<summary><b>Error Logs</b></summary>\n\n{res.text}\n\n</details>\n\n"
    return final_response

iface1 = gr.Interface(ingest_book, inputs=[gr.File(label="Upload one or more books (PDF and DOCX allowed)", file_count="multiple", file_types=[".pdf", ".PDF", ".docx", ".DOCX"])], outputs=[gr.Textbox(label="Ingestion Status")])
iface2 = gr.ChatInterface(fn=search_library)
iface3 = gr.ChatInterface(fn=search_web)
iface = gr.TabbedInterface([iface1, iface2, iface3], ["Upload your books📚", "Search your e-library🔍", "Search the web for new books🌍"], title="E-Library Agent", theme=gr.themes.Citrus(primary_hue="indigo", secondary_hue="teal"))
app = gr.mount_gradio_app(app=app, blocks=iface, path="")

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
