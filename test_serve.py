#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langserve import add_routes
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,# Verbose is required to pass to the callback manager
)

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    llm,
    path="/llamacpp",
)



model = llm
prompt_chatbot = PromptTemplate(
    input_variables=["question","chat_history"],
    template="""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>
Here is the chat history:{chat_history}
Answer the following question in one sentence:{question} [/INST]
""",
)
memory = ConversationBufferWindowMemory(k=5,memory_key="chat_history")
conversation = LLMChain(llm=llm, prompt=prompt_chatbot, memory=memory)
add_routes(
    app,
    conversation,
    path="/chatbot",
)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)