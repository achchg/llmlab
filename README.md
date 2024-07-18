# llmlab - my LLM Lab (WIP)

## Conda env setup
``` 
conda env create -f environment.yml
```

## Simple example scripts running LLM locally (on my laptop)
- `gpt4all`
- `ollama_code.py`: My choice for further experiment.


# A CV summarization chatbot
- `/frontend`: Frontend implementattion
  - `app_gradio.py`: A gradio app implementation using Ollama models.
    - `outer_evaluate`: evaluation method for meta-step (outer-loop)
    - `task`: define task (support/query)
    - `get_task_dataset`: get support and query datasets and make embeddings
    - `inner_loop_train`: MAML inner-loop gradient descent update
    - `inner_loop_query`: define query set for each task and calculate the gradient of loss with query set
    - `inner_loop`: the overall inner_loop step
- `/src`: Backend implementation
  - `embeddings`: 
    - `data_extraction.py`: 
      - `extract_text_from_url` extracts texts from the url (e.g. my personal website and the urls)
      - `generate_embeddings` generates embeddings with Huggingface's SentenceTransformerEmbeddings and saves the embeddings as a vector DB with PGVector
  - `prompts`:
    - `prompt_template.py`: for prompt management
      - `CVPrompt` is the basic template for resume summarization

- To run the app: `python frontend/app_gradio.py --model_name mistral`
- Demo: <video src='https://www.youtube.com/watch?v=eSxMgvQcgqo' width=180/>

