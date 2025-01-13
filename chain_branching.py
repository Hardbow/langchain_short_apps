from dotenv import load_dotenv
from os import getenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()
model = ChatOpenAI(model=getenv("SMART_LLM"), api_key=getenv("OPENAI_API_KEY"))

positive_messages = [
    ("system", "You are a helpful assistant."),
    ("human", "Generate a thank you note for this positive feedback: {feedback}.")
]
positive_feedback_template = ChatPromptTemplate(messages=positive_messages)

negative_messages = [
    ("system", "You a helpful assistant."),
    ("human", " Generate a response addressing this negative feedback: {feedback}.")
]
negative_feedback_template = ChatPromptTemplate.from_messages(negative_messages)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You a helpful assistant"),
        ("human", "Generate a request for more details for this neural feedback: {feedback}.")
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You a helpful assistant"),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}.")
    ]
)

print(type(positive_feedback_template),type(negative_feedback_template) ,type(neutral_feedback_template), sep='\n')

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You a helpful assistant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.")
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    neutral_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

# "The product is excellent. I really enjoyed using it and found it very helpful."
# "The product is terrible. It broke after just one use and the quality is very poor."
# "The product is okay. It works as expected but nothing exceptional."
# "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "The whole family used this product. Now we will sell the apartment and buy more of this product. We showed it to the neighbors, they already sold their car and bought this product with all the money"
result = chain.invoke({"feedback": review})
print(result)