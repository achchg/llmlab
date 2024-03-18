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

llm_chain.run(question)