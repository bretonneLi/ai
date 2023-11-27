from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma,FAISS
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore


llm = LlamaCpp(
    model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf",n_ctx=2048
)
loader = WebBaseLoader("http://tingyimiao.com/")
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

#Locally store the embeddings
underlying_embeddings = LlamaCppEmbeddings(model_path="/home/dubenhao/llama/llama-2-7b/ggml-model-q4_k_m.gguf")
fs = LocalFileStore("/home/dubenhao/test_cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, fs,
)
db = FAISS.from_documents(all_splits, cached_embedder)



memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)

retriever = db.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

print(qa("What is Tingyimiao"))
