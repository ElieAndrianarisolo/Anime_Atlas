import os
import pysrt
from dotenv import load_dotenv
import re
from concurrent.futures import ThreadPoolExecutor
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document class to handle text documents and optional metadata.
class Document:
    """ Represents a document with content and optional metadata. """
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Main function to process and embed subtitle files.
def files_embedding() :

    load_dotenv()

    # Function to load and clean subtitle text from .srt files.
    def load_and_process_subtitle(file_path):
        """ Load subtitle files, clean their text, and return cleaned text. """
        try:
            subs = pysrt.open(file_path, encoding='utf-8')
            cleaned_text = "\n".join(re.sub('<.*?>', '', sub.text).strip() for sub in subs)
            return re.sub(r'\n+', '\n', cleaned_text).strip()
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            return None

    # Function to load texts from all .srt files in a directory.
    def load_texts_from_directory(directory):
        """ Walk through a directory and process all .srt files. """
        texts = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".srt"):
                    file_path = os.path.join(root, filename)
                    text = load_and_process_subtitle(file_path)
                    if text:
                        texts.append(text)
        return texts

    # Function to process texts in parallel, splitting them into smaller chunks.
    def parallel_process_texts(texts):
        """ Use parallel processing to split texts into smaller chunks for better handling. """
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        with ThreadPoolExecutor() as executor:
            chunks = list(executor.map(splitter.split_text, texts))
        return [item for sublist in chunks for item in sublist]

    # Prompt template for contextualizing questions based on chat history.
    all_texts = load_texts_from_directory('static/subtitles')
    documents = [Document(content) for content in parallel_process_texts(all_texts)]
    vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
    
    return vectorstore

# Prompt template for answering questions using retrieved context.
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

# Function to retrieve or create a chat message history for a session.
qa_system_prompt = """"You are an assistant specialized in anime and manga. \
Review the question:
- If it mentions specific anime or manga or seeks information about them, use your knowledge and the provided context to give a concise, informative answer.
- If the question is about identifying an anime or manga based on described characteristics (e.g., genre, plot elements), provide a recommendation based on your knowledge and the provided context.
- If the question does not relate to anime or manga, simply respond that you do not have the answer.

Limit your response to three sentences and avoid mentioning the context in your answer. Strive for a human-like and engaging response.

{context}"""
