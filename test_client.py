from langserve import RemoteRunnable



conversation = RemoteRunnable("http://localhost:8000/chatbot/")

print(conversation.invoke(
    {"question": "Translate this sentence from English to French: I love programming."}
))
print(conversation.invoke({"question": "Translate it to German."}))
print(conversation.invoke({"question": "Translate it to Japanese."}))
print(conversation.invoke({"question": "Translate it to Chinese."}))
print(conversation.invoke({"question": "Translate this sentence from English to French: I love you."}))

