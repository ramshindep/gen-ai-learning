from langchain_ollama import ChatOllama
from lngchain_core.messages import SystemMessage,HumanMessage,AIMessage

llm=ChatOllama(
    model="gemma:2b",
    max_tokens=100,
    temperature=0.7,
    n=2
)

messages=[]

while True:
    prompt=input("User:")
    if prompt.lower()=="exit":
        break
    
    
    messages.append(HumanMessage(prompt))
    response=llm.invoke(messages)
    messages.append(AIMessage(response.content))
    
    print("Bot:"+response.content)
    
    print("---------------------------------------------------------------------------------------")