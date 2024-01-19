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
from langchain.embeddings import LlamaCppEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
from langchain_community.vectorstores import Chroma
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Chroma.from_documents(documents=all_splits, 
                                    collection_name="rag-chroma",
                                    embedding=LlamaCppEmbeddings(model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf"),
                                    )
retriever = vectorstore.as_retriever()
template="""[INST]
    Use the following pieces ofcontext to answer the question.If no context provided, answer like a AI assistant. 
    
    {context} 
    Question: {question}
    [/INST]
"""


'''#
vectorstore=FAISS.load_local("/home/dubenhao/vectorstore/db_faiss", embedding=LlamaCppEmbeddings(model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf"))
retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )
'''
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever,     
    chain_type_kwargs={
        "prompt":  PromptTemplate(
            template=template,
            input_variables=["context", "question"]),
    }
)
add_routes(
    app,
    qa_chain,
    path="/qa_chain",
)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)