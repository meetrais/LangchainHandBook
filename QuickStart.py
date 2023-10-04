from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = OpenAI(openai_api_key="sk-CopyYourOpenAIKeyHere")

chat_model=ChatOpenAI(openai_api_key="sk-CopyYourOpenAIKeyHere")
text = "What would be a good company name for a company that makes colorful socks?"

llm_out =  llm.predict(text)
chat_out =  chat_model.predict(text)

print("llm_out=" + llm_out.strip())
print("chat_out=" + chat_out.strip())

messages = [HumanMessage(content=text)]

llmMessages =  llm.predict_messages(messages)
ChatMessages=  chat_model.predict_messages(messages)

print(llmMessages)
print(ChatMessages)







