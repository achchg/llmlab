import requests
from bs4 import BeautifulSoup
import time
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.pgvector import PGVector


from logging import getLogger
logger = getLogger(__name__)

# use requests to fetch and extract page content
def get_page_content(url):
    """
    Retrieves the content of a webpage given a URL.

    Args:
        url (str): The URL of the webpage to retrieve.

    Returns:
        BeautifulSoup: A BeautifulSoup object representing the parsed HTML content of the webpage.

    Raises:
        Exception: If the webpage content cannot be retrieved after multiple attempts.

    Notes:
        - The function uses the `requests` library to send a GET request to the URL with a custom User-Agent header.
        - If the response status code is 200, the function returns a BeautifulSoup object parsed from the response content.
        - If the response status code is 429 (rate limited), the function waits for the specified number of seconds and retries.
        - If the response status code is anything other than 200 or 429, an exception is raised.
        - The function retries a maximum of 5 times before raising an exception if the content cannot be retrieved.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    max_retries = 5
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            logger.info("url Content is available.")
            return BeautifulSoup(response.content, 'html.parser')
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))  # Default to 5 seconds if not specified
            logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            logger.exception(f"Failed to fetch page content, status code: {response.status_code}.")
            raise Exception(f"Failed to fetch page content, status code: {response.status_code}.")
        
        logger.exception(f"Max retries exceeded for URL: {url}.")
    raise Exception(f"Max retries exceeded for URL: {url}.")


# find and extract other urls
def extract_urls(url):
    """
    Retrieves additional external URLs from the profile URL.
    
    Args:
        url (str): The URL of the webpage containing the external URLs.
    
    Returns:
        list: A list of extracted external URLs.
    """

    soup = get_page_content(url)
    urls = []
    titles = ['LinkedIn', 'Google Scholar']
    logger.info("Start extracting additional external urls from the profile url.")
    for a_tag in soup.find_all('a', href=True):
        if 'title' in a_tag.attrs and a_tag['title'] in titles:
            url = a_tag['href']
            if url.startswith('http'):
                urls.append(url)
    
    logger.info("Extraction completed.")
    return urls

# extract text from urls
def extract_text_from_url(url):
    """
    Retrieves text content from a webpage URL.

    Args:
        url (str): The URL of the webpage to extract text from.

    Returns:
        str: The extracted text content from the webpage.
    """
    soup = get_page_content(url)
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

# generate embedding
def generate_embeddings(text_list, meta_data_list, new_docs=True):
    """
    Generates embeddings with PGVector for a list of texts and metadata using SentenceTransformer model.
    
    Args:
        text_list (list): List of texts to generate embeddings for.
        meta_data_list (list): List of metadata corresponding to the texts.
        new_docs (bool, optional): If True, creates a new embedded database; else adds data to an existing one. Defaults to True.
    
    Returns:
        None
    """
    embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    
    CONNECTION_STRING=PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host="localhost",
        port="6024",
        database="postgres",
        user="langchain",
        password="langchain")
    collection_name = "embeddings"

    if new_docs:
        db = PGVector.from_texts(
            texts=text_list,
            metadatas=meta_data_list,
            connection_string=CONNECTION_STRING,
            collection_name=collection_name,
            embedding=embedding_function,
            pre_delete_collection=True)
        logger.info(f"New embedded db {collection_name} is created.")

    else:
        db = PGVector(
            connection_string=CONNECTION_STRING,
            collection_name=collection_name,
            embedding_function=embedding_function)
        new = db.add_texts(db)
        logger.info(f"Added additional data point to the embedded db {collection_name}.")
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new", type=bool, help="create new collection"
    )

    args = parser.parse_args()
    url = 'https://achchg.github.io'

    # find and extract other urls
    embedded_urls = extract_urls(url)

    # uncomment the below when the rate limit is hit
    # embedded_urls = []
    urls = [url] + embedded_urls

    # extract text from urls
    text_list = []
    metadata_list = []
    for url in urls:
        text = extract_text_from_url(url)
        if len(text) > 0:
            text_list.append(text)
            metadata_list.append({"url":url})
        else:
            logger.info(f"No text found in {url}.")

    # generate embeddings
    generate_embeddings(text_list, metadata_list, new_docs=args.new)
    