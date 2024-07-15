import argparse

import gradio as gr

from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import TextIteratorStreamer, AutoTokenizer
import torch
from threading import Thread
import ollama
import os


from src.prompts.prompt_template import CVPrompt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type = str,
    default="ollama3", #ollama3
    required=False,
    help="Model Name to pull model"
)

args = parser.parse_args()

# model = AutoModelForCausalLM.from_pretrained(
#     args.model_name,
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.float16
# )

tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')

CONNECTION_STRING=PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host="localhost",
        port="6024",
        database="postgres",
        user="langchain",
        password="langchain")

embedding_function = SentenceTransformerEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

collection_name = "embeddings"
db = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=collection_name,
    embedding_function=embedding_function)

with gr.Blocks() as demo:
    gr.HTML(
        f"""
        <h3> Model Name: {args.model_name}
    """
    )

    chat_history = gr.Chatbot(label="QA Bot")
    msg = gr.Textbox(label="Question", interactive=True)

    with gr.Accordion(label="Generation Parameters", open = False):
        prompt = CVPrompt()
        prompt_format = gr.Textbox(
            label="Formatting prompt",
            value=f"""{prompt.template}
""", interactive=True
        )
        with gr.Row():
            max_new_tokens = gr.Number(
                minimum=100, maximum=500, value = 100, label = "Max New Tokens", interactive=True
            )
            temperature = gr.Slider(
                minimum=0, maximum=1.0, value = 0.1, label="Temperature", interactive=True
            )
            k = gr.Slider(
                minimum=1, maximum=5, value = 1, label="# of Retrieved files", interactive=True
            )

    
    clear = gr.ClearButton([msg, chat_history])
    


    def user(user_message, history):
        if history:
            history.append([user_message, None])
            return "", history
        else:
            return "", [[user_message, None]]
    
    def format_history(history: list[list[str, str]], system_prompt: str):
        chat_history = [{"role": "system", "content": system_prompt}]
        print(f"history:{history}")
        for query, response in history:
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response})  
        # chat_history.append({"role": "user", "content": msg})
        return chat_history

    def bot(chat_history, temperature, k=1):
        print(f"chat_history-start: {chat_history}")
        msg = chat_history[-1][0]
        similarity_scores = db.similarity_search_with_score(msg, k = k)

        print(f"content-all: {similarity_scores}")
        print(f"content: {similarity_scores[0][0].page_content}")
        print(f"similarity_score: {similarity_scores[0][1]}")

        formatted_prompt = prompt.template.format(
            context = similarity_scores[0][0].page_content
        )
        
        chat_history = format_history(chat_history, formatted_prompt)
        print(f"chat_history: {chat_history}")

        generation_kwargs = dict(
            model=args.model_name,
            # prompt = prompt.template,
            stream=True,
            # max_new_tokens=max_new_tokens,
            # do_sample=True,
            temperature=temperature,
            # use_cache=True
        )
        
        response = ollama.chat(model=args.model_name, stream=False, messages=chat_history)['message']['content']
        print(response)

        print(f"chat_history - 0: {chat_history}")
        chat_history[-1]["content"] = ""
        print(f"list of responses: {list(response.split(' '))}")
        for new_text in list(response.split(' ')):
            if new_text != " ":
                # print(f"new_text: {new_text}")
                chat_history[-1]["content"] += new_text + " "
                # print(f"chat_history - 1: {chat_history}")

        chat_history_no_prompt = chat_history[1:]
        chat_history = [[chat_history_no_prompt[i*2]['content'], 
                         chat_history_no_prompt[i*2+1]['content']] for i in range(len(chat_history_no_prompt) // 2)]
        
        print(f"chat_history - 2: {chat_history}")
        yield chat_history
    

    msg.submit(user, [msg, chat_history], [msg, chat_history], queue=False).then(
        bot, [chat_history, temperature], chat_history
    )



demo.queue()
demo.launch()
    


