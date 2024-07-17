import argparse

import gradio as gr

from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import ollama

from src.prompts.prompt_template import CVPrompt
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# use parser to pull model
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type = str,
    default="ollama3", #ollama3
    required=False,
    help="Model Name to pull model"
)

args = parser.parse_args()
langfuse = Langfuse()
langfuse_handler = CallbackHandler(
    session_id="test-1234",
    user_id = "chi-local"
)

# connect to vectorstore
CONNECTION_STRING=PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host="localhost",
        port="6024",
        database="postgres",
        user="langchain",
        password="langchain")

# select the same embedding model that we used for vectorstore creation
embedding_function = SentenceTransformerEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

collection_name = "embeddings"
db = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=collection_name,
    embedding_function=embedding_function)

# set up gradio
with gr.Blocks() as demo:
    gr.HTML(
        f"""
        <h3> Model Name: {args.model_name}</h3>"""
    )

    # create chatbot
    chat_history = gr.Chatbot(label="QA bot for my portfolio")
    msg = gr.Textbox(label="Question", interactive=True)

    # create interactive hyper-parameters
    with gr.Accordion(label="LLM Hyper Parameters", open = False):
        prompt = CVPrompt()
        prompt_format = gr.Textbox(
            label="Formatting prompt",
            value=f"""{prompt.template}""", interactive=True
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
        """
        A function that handles user messages in a chatbot system.

        Args:
            user_message (str): The message sent by the user.
            history (list): The history of the chat conversation.

        Returns:
            tuple: A tuple containing an empty string and the updated chat history.
        """

        if history:
            # Append the user's message to the history
            history.append([user_message, None])
            return "", history
        else:
            return "", [[user_message, None]]
    
    def format_history(history: list[list[str, str]], system_prompt: str):
        """
        Format the given chat history into a list of dictionaries representing each message in the conversation.

        Args:
            history (list[list[str, str]]): The chat history, where each sublist contains a query and its corresponding response.
            system_prompt (str): The initial system prompt to be added to the chat history.

        Returns:
            list[dict]: The formatted chat history, where each dictionary represents a message and contains the role and content of the message.
        """
        chat_history = [{"role": "system", "content": system_prompt}]
        print(f"history:{history}")
        for query, response in history:
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": response})  
        return chat_history

    def bot(chat_history, temperature, k, num_predict):
        """
        A function that processes chat history and generates a response using the Ollama chat model.
        
        Args:
            chat_history (list): The history of the chat conversation.
            temperature (float): The temperature parameter for response generation.
            k (int): The number of similarity scores to consider.

        Returns:
            generator: A generator that yields the processed chat history.
        """

        print(f"chat_history-start: {chat_history}")
        msg = chat_history[-1][0]
        similarity_scores = db.similarity_search_with_score(msg, k = k)

        print(f"content-all: {[doc[0].page_content for doc in similarity_scores]}")
        print(f"content-first: {similarity_scores[0][0].page_content}")
        print(f"similarity_score: {similarity_scores[0][1]}")

        formatted_prompt = prompt.template.format(
            context = ' \n ==='.join([doc[0].page_content for doc in similarity_scores])
        )
        print(f"formatted_prompt: {formatted_prompt}")

        langfuse.create_prompt(
            name="ollama-cv-prompt",
            prompt=formatted_prompt,
            is_active=True,
            config = {
                "model": args.model_name,
                "temperature": temperature,
                "k": k,
                "supported_languages": ["en"]
            }
        )
        chat_history = format_history(chat_history, formatted_prompt)
        print(f"chat_history: {chat_history}")

        generation_kwargs = dict(
            model=args.model_name,
            stream=False,
            messages=chat_history,
            options = {"temperature": temperature, "k": k},
        )
        
        response = ollama.chat(**generation_kwargs)['message']['content']
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
        bot, [chat_history, temperature, k], chat_history
    )


demo.queue()
demo.launch()
    


