import requests
from bs4 import BeautifulSoup
import time
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.pgvector import PGVector


from logging import getLogger
logger = getLogger(__name__)

# use requests to fetch and extract page content
def get_page_content(url):
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
    soup = get_page_content(url)
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

# generate embedding
def generate_embeddings(text_list, meta_data_list, new_docs=True):
    model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    embedding_function = SentenceTransformerEmbeddings(
    model_name='all-MiniLM-L6-v2')
    
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
    

url = 'https://achchg.github.io'
embedded_urls = extract_urls(url)
# embedded_urls = []
urls = [url] + embedded_urls

text_list = []
metadata_list = []
for url in urls:
    text_list.append(extract_text_from_url(url))
    metadata_list.append({"url":url})

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new", type=bool, help="create new collection"
    )

    args = parser.parse_args()
    generate_embeddings(text_list, metadata_list, new_docs=args.new)
    