from dotenv import load_dotenv
from os import getenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI


def get_information(model_output):
    print("We get this features:\n", model_output)
    return model_output

def analyze_pros(features):
    pros_template = ChatPromptTemplate(
        [
            ("system", "You are an expert product reviewer." ),
            ("human", "Given these features: {features}, list the pros of these features"),
        ]
    )
    return pros_template.format_prompt(features=features)

def analyze_cons(features):
    cons_template = ChatPromptTemplate(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the cons of these features.")
        ]
    )
    return cons_template.format_prompt(features=features)

def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons\n{cons}"

load_dotenv()
model = ChatOpenAI(model=getenv("SMART_LLM"), api_key=getenv("OPENAI_API_KEY"))
messages = [
    ("system", "You are an expert product reviewer."),
    ("human", "List the main features of the product {product_name}")
]
prompt_template = ChatPromptTemplate.from_messages(messages)

pros_branch = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | get_information
    | RunnableParallel(branches={"pros": pros_branch, "cons": cons_branch})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name": "DVD"})

print(result)