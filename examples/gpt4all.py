from langchain_community.llms.gpt4all import GPT4All
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

local_path = ('data/gpt4all-falcon-newbpe-q4_0.gguf.bin')
llm = GPT4All(model=local_path)

template = """{question}"""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What would be a good time to eat lunch?"

llm_chain.run(question)