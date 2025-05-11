from pydantic import BaseModel, model_validator
from llama_index.core.llms import ChatMessage
from typing import List, Optional, Self

class ChatHistory(BaseModel):
    messages: Optional[List[ChatMessage]] = None
    @model_validator(mode="after")
    def validate_messages(self) -> Self:
        if self.messages is None:
            self.messages = []
        return self
