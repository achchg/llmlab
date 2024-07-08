import argparse

import gradio as gr

from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import TextIteratorStreamer, AutoTokenizer
import torch
from threading import Thread
import ollama
import os


from src.prompts.prompt_template import TravelPrompt


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
        prompt = TravelPrompt()
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
        return f"Response: {user_message}", [[user_message, None]]

    def bot(chat_history, prompt_format, max_new_tokens, temperature, k=1):
        similarity_scores = db.similarity_search_with_score(chat_history[-1][0], k = k)
        
        formatted_inst = prompt_format.format(
            context = similarity_scores[0][0].page_content,
            question = chat_history[-1][0]
        )
        print(f"content: {similarity_scores[0][0].page_content}")
        print(f"similarity_score: {similarity_scores[0][1]}")

        input_ids = tokenizer(
            formatted_inst, return_tensors="pt", truncation=True
        )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

        generation_kwargs = dict(
            model=args.model_name,
            prompt = prompt.template,
            stream=streamer,
            # max_new_tokens=max_new_tokens,
            # do_sample=True,
            # temperature=temperature,
            # use_cache=True
        )

        print(ollama.generate(model=args.model_name,
            prompt = prompt.template,
            stream=streamer))

        thread = Thread(target=ollama.generate, kwargs=generation_kwargs)
        thread.start()
        chat_history[-1][1] = ""
        for new_text in streamer:
            chat_history[-1][1] += new_text
            yield chat_history

    msg.submit(user, [msg, chat_history], [msg, chat_history], queue=False).then(
        bot, [chat_history, prompt_format, max_new_tokens, temperature], chat_history
    )



demo.queue()
demo.launch()
    


