from linkup import LinkupClient, LinkupSourcedAnswer
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from typing import Tuple
import json

with open("/run/secrets/mistral_key", "r") as f:
    openai_api_key = f.read()
f.close()

with open("/run/secrets/linkup_key", "r") as g:
    linkup_api_key = g.read()
g.close()

linkup_client = LinkupClient(api_key=linkup_api_key)

class EvaluateContext(BaseModel):
    context_is_ok: int = Field(description="Is the context relevant to the question? Give a score between 0 and 100")
    reasons: str = Field(description="Explanations for the given evaluation")

class BookInfo(BaseModel):
    title: str = Field(description="Title of the book")
    author: str = Field(description="Author of the book")
    year: int = Field(description="Publication year")
    summary: str = Field(description="Summary of the book's plot")

llm = OpenAI(model="gpt-4.1-2025-04-14", api_key=openai_api_key)
llm_eval = llm.as_structured_llm(EvaluateContext)

async def deepsearch(query: str) -> str:
    """Useful to search for precise information in the depths of the web when the users asks you information about books of a certain type, or about a specific book.
    
    Args:
        query (str): the query to be searched"""
    response = linkup_client.search(
        query=query,
        depth="deep",
        output_type="structured",
        structured_output_schema=BookInfo,
    )
    answer = response.model_dump_json(indent=4)
    return answer


async def evaluate_context(original_prompt: str = Field(description="Original prompt provided by the user"), context: str = Field(description="Contextual information from the web")) -> str:
    """
    Useful for evaluating the relevance of retrieved contextual information in light of the user's prompt.

    This tool takes the original user prompt and contextual information as input, and evaluates the relevance of the contextual information. It returns a formatted string with the evaluation scores and reasons for the evaluations.

    Args:
        original_prompt (str): Original prompt provided by the user.
        context (str): Contextual information from the web.
    """
    messages = [ChatMessage.from_str(content=original_prompt, role="user"), ChatMessage.from_str(content=f"Here is some context that I found that might be useful for replying to the user:\n\n{context}", role="assistant"), ChatMessage.from_str(content="Can you please evaluate the relevance of the contextual information (giving it a score between 0 and 100) in light or my original prompt? You should also tell me the reasons for your evaluations.", role="user")]
    response = await llm_eval.achat(messages)
    json_response = json.loads(response.message.blocks[0].text)
    final_response = f"The context provided for the user's prompt is {json_response['context_is_ok']}% relevant.\nThese are the reasons why you are given these evaluations:\n{json_response['reasons']}"
    return final_response
