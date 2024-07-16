from logging import getLogger
import os
from langchain.prompts import PromptTemplate

__all__ = ["CVPrompt"]

class CVPrompt:
    """
    CVPrompt class to manage the prompt template used for the CV usecase.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.template = """You are a chatbot that supports talent acquisition. Please provide a summary of the profile in terms of the professional experience you received, highlighting key achievements and responsibilities. 
        profile content: {context}
        """