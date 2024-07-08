from logging import getLogger
import os
from langchain.prompts import PromptTemplate

__all__ = ["TravelPrompt"]

class TravelPrompt:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.template = """You are a travel agent that help people prepare travel itinerary. 
        {question}
        """