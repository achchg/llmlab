from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# llm = Ollama(model="llama2")
llm = ChatOllama(model="llama3")

text = "You are a travel agent that help people prepare travel itinerary. {question}"
prompt = ChatPromptTemplate.from_template(text)
chain = prompt | llm | StrOutputParser()

# messages = [HumanMessage(content=text)]
# print(llm.invoke(text))

langfuse = Langfuse()
langfuse.create_prompt(
    name="ollama-test-prompt",
    prompt=text,
    is_active=True,
    config = {
        "model": "llama2",
        "temperature": 0.2,
        "supported_languages": ["en"]
    }
)
langfuse_handler = CallbackHandler(
    session_id="test-1234",
    user_id = "chi-local"
)

print(chain.invoke({"question": "Travel plan for 6 days Iceland travel in June."}, config={"callbacks": [langfuse_handler]}))