from logging import getLogger
import os
from langchain.prompts import PromptTemplate

__all__ = ["CVPrompt"]

class CVPrompt:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.template = """You are an expert in talent acquisition. Please provide a summary of the profile in terms of the professional experience you received, highlighting key achievements and responsibilities. 
        content: {context}
        """