from dotenv import load_dotenv
from os import getenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model=getenv("SMART_LLM"), api_key=getenv("OPENAI_API_KEY"))

message = [
    SystemMessage(content="Give answer on the following question"),
    HumanMessage(content="Who is the greatest film director?"),
    AIMessage(content="Ultimately, the greatest film director is a matter of personal opinion and can differ widely among film enthusiasts."),
    HumanMessage(content="Nevertheless, who is the greatest film director making the most significant movies?")
]

result = model.invoke(message)

print(f"Answer:\n{result.content}")