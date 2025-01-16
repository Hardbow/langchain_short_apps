from dotenv import load_dotenv
import os
from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
file_name = "hobbit.txt"
file_dir = "books"

project_data_folder = Path(os.path.dirname(os.path.abspath(__file__)))
file_to_open = project_data_folder / file_dir / file_name
persist_directory = project_data_folder / "faiss_index"
print(file_to_open)
print(persist_directory)

if not os.path.exists(persist_directory):
    print("Vector folder does not exist. Initializing vector store...")

    if not os.path.exists(file_to_open):
        raise FileNotFoundError(f"The file {file_to_open} does not exist. Please check the path.")

    loader = TextLoader(file_to_open, encoding='UTF-8')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=[" ", ",", "\n"])
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunk Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[5].page_content}\n")

    print("\n---- Creating embeddings ----")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    print("\n---- Finished creating embeddings ----")

    print("\n---- Creating vector store -----")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("\n---- Finish creating vector store -----")

else:
    print("Vector store already exists. No need to initialize.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

query = input(f"Your query:\n")

retrieved_docs = vectorstore.similarity_search(query, k=3)

# retrieved_docs = vectorstore.similarity_search_with_score(query="Как звали троллей", search_type="mmr", search_kwargs={"k": 10, "score_threshold": 0.99})
# retrieved_docs = retriever.invoke("Как звали троллей")
# similarity_search_with_relevance_scores - use cosine similarity

for num, item in enumerate(retrieved_docs):
    print(f"\n---{num + 1}----\n{item.page_content}")
