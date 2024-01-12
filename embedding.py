from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import LlamaCppEmbeddings
loader =  PyPDFLoader("/home/dubenhao/tingyimiao.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
splits = text_splitter.split_documents(data)
vectorstore = FAISS.from_documents(documents=splits, embedding=LlamaCppEmbeddings(model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf"))
vectorstore.save_local("/home/dubenhao/vectorstore/db_faiss")

template="""[INST]
    Use the following pieces ofcontext to answer the question.If no context provided, answer like a AI assistant. 
    Answer the question in Chinese.
    {context} 
    Question: {question}[/INST]
"""