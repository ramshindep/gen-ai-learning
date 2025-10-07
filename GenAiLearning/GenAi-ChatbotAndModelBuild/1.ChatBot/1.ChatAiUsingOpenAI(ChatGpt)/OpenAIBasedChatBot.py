import os
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o",max_tokens=100)

prompt="What is the capital of India?"

#Call the OpenAi ApI

response =llm.invoke(prompt)
print(response)