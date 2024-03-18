from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

llm = Ollama(model="llama2")
chat_model = ChatOllama()

text = "What would be a good time to eat lunch?"
messages = [HumanMessage(content=text)]

print(llm.invoke(text))