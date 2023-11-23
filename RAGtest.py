import bs4
from langchain import hub
from langchain.llms import LlamaCpp
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=LlamaCppEmbeddings(model_path="/home/dubenhao/llama/llama-2-7b/ggml-model-q4_k_m.gguf"))
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = LlamaCpp(model_path="/home/dubenhao/llama/llama-2-7b/ggml-model-q4_k_m.gguf", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
for chunk in rag_chain.stream("What is Task Self-Reflection?"):
    print(chunk, end="", flush=True)