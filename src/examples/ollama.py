from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

llm = Ollama(model="llama2")
chat_model = ChatOllama()

text = "What would be a good time to eat lunch?"
messages = [HumanMessage(content=text)]

llm.invoke(text)

# chat_model.invoke(messages)

from langchain_community.llms import GPT4All
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

local_path = 'path to where you save the model bin file'
llm = GPT4All(model=local_path, callbacks=[StreamingStdOutCallbackHandler()])

template = """{question}"""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What would be a good time to eat lunch?"

print(llm_chain.run(question))