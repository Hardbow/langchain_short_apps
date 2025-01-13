from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# template = "Tell me a joke about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)
#
# print("------------Prompt from Template-------------")
# prompt = prompt_template.invoke({"topic": "cats"})
# print(prompt)

# template_multiple = """You are a helpful assistant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant:"""
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
# print("------------Prompt with multiple Placeholders-------------")
# print(prompt)

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})

print("------------Prompt with System and Human Messages (Tuple)-------------")
print(prompt)