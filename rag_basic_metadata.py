from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
import dotenv
import os
import logging
import warnings

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langsmith import traceable

from models import Query
from utils import (
    orchestrator,
    split_into_paragraphs,
    generate_answer,
    summarize_history,
    hallucination_check,
    refine_query,
    split_into_paragraphs

)

dotenv.load_dotenv()

# Suppress the FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(filename="app.log",encoding='utf-8'), # Logs will be saved in app.log
        logging.StreamHandler() #Contitue showing logs in the console
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("Initialize logger")

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup for embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


# Initialize FAISS vector store
def initialize_vectorstore(file_name: str):
    if os.path.exists("faiss_index"):
        logger.info("FAISS vectorstore loading from local index has started.")
        vectorstore = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        logger.info("FAISS vectorstore loaded from local index.")
        return vectorstore
    else:
        try:
            logger.info('os.path.exists("faiss_index") NOT exist.')
            with open(f"{file_name}", "r", encoding="utf-8") as file:
                text = file.read()
                paragraphs = split_into_paragraphs(text)
                vectorstore = FAISS.from_texts(paragraphs, embeddings)
                vectorstore.save_local("faiss_index")

        except FileNotFoundError:
            logger.error("File hobbit.txt not found")
            raise
        except Exception as e:
            logger.exception("Failed to initialize FAISS vectorstore")
            raise

vectorstore = initialize_vectorstore("hobbit.txt")

@traceable(name='vector_search')
def vector_search(query: Query, vectorstore):
    logger.info("Vector search required. Performing similarity search.")
    retrieved_docs = vectorstore.similarity_search(query.question, k=5)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    logger.info(
        f"Retrieved context from vector search: {context[:100]}..."
    ) # log first 100 chars
    return context


@app.post("/query")
async def start_point(query: Query):
    """
    FastAPI endpoint that calls the traceable function.
    """
    try:
        response = await query_text_base(query)
        return response
    except Exception as e:
        logger.exception("An error ocurred while function start_point was proccessing")
        raise HTTPException(status_code=500, detail=str(e))


@traceable(name="query_text_base")
async def query_text_base(query: Query):
    if vectorstore is None:
        logger.error("Vectorstore not initialized")
        raise HTTPException(status_code=500, detail="Vector not initialized")

    try:
        logger.info(f"Received query: {query}, type(query) {type(query)}, question - {query.question}")
       # Step 1: Determine if vector search is need
        use_vector, model_use = orchestrator(query)
        logger.info(f"use_vector - f{use_vector}, model_use - {model_use}")
        if use_vector:
            context = vector_search(query, vectorstore)
            logger.info(f"Function vecrtor_search has returned context {context[:100]}")

        else:
            logger.info("Vector search not required. Proceeding without context.")
            context = ""
        query.context = query.context + [context]

        # 2. Use LLM to generate an answer based on retrieved documents
        answer = generate_answer(query.question, context, query.history, model_use)
        logger.info(f"Function generate_answer has returned answer {answer}")

        # 3. Chech for hallucinations
        if hallucination_check(query.question, answer, context):
            logger.warning("Hallucination detected in the answer. Refining the query.")
            query.question = refine_query(query.question, answer)
            return await query_text_base(query)

        # 4. Prepare response with conversation history
        update_history = query.history.copy()
        update_history.append({"role": "user", "content": query.question})
        update_history.append({"role": "assistant", "content": answer})

        # Optimize update_history upto 2000 messages
        if len(update_history) > 2000:
            logger.info("Conversation history too long. Prune it")
            response_history = summarize_history(update_history)
        else:
            response_history = update_history
        return {"answer": answer, "history": response_history}

    except Exception as e:
        logger.exception("An error occurred while processing the query")
        raise HTTPException(status_code=500, detail=str(e))


# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1:8000",
    "http://192.168.0.160:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
