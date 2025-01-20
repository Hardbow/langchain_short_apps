from dotenv import load_dotenv
import os
import logging
import warnings

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Suppress the FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(filename="app.log", encoding='utf-8'),  # Logs will be saved in app.log
        logging.StreamHandler()  # Continue showing logs in the console
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("Initialize logger")

app_dir = Path(os.path.dirname(os.path.abspath(__file__)))
books_dir = app_dir / "books"
faiss_meta_dir = app_dir / "faiss_index" / "FAISS_index_metadata"

logger.info(f"Books directory: {books_dir}")
logger.info(f"FAISS vectorstore with metadata directory: {faiss_meta_dir}")

# Setup for embeddings
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=os.getenv("OPENAI_API_KEY"))

if not os.path.exists(faiss_meta_dir):
    logger.info(f"The directory {faiss_meta_dir} does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path"
        )

    # List all text files in the directory
    book_files = [file for file in os.listdir(books_dir) if file.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = books_dir / book_file
        loader = TextLoader(file_path, encoding='utf-8')
        book_doc = loader.load()
        documents.append(book_doc[0])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    logger.info("------- Document Chunk Information ---------------")
    logger.info(f"Number of document chunk: {len(docs)}")

    logger.info("----- Creating embeddings ----------- ")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("------ Finished creating embeddings ---------")

    logger.info("------- Creating vector store ---------")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_meta_dir)

else:
    # Load the existing vector store with embedding function
    logger.info(f"Load vector store data from {faiss_meta_dir}")
    vectorstore = FAISS.load_local(faiss_meta_dir, embeddings, allow_dangerous_deserialization=True)

query = 'Как умер Вини-Пух'
# Retrieve relevant documents based on the query
retrieved_docs = vectorstore.similarity_search_with_score(query, k=5)

for i, cont in enumerate(retrieved_docs):
    print(f"\n-----{i}------\n{cont[0].page_content}")