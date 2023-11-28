import bs4
from langchain.llms import LlamaCpp
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,# Verbose is required to pass to the callback manager
)
prompt = PromptTemplate(
    input_variables=["question","chat_history"],
    template="""<s>[INST] <<SYS>>
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
<</SYS>>
{context}
Question:{question} [/INST]
""",
)
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

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
underlying_embeddings = LlamaCppEmbeddings(model_path="/home/dubenhao/llama/llama-2-7b/ggml-model-q4_k_m.gguf")
fs = LocalFileStore("/home/dubenhao/cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, fs,
)

vectorstore = Chroma.from_documents(splits,cached_embedder)
retriever = vectorstore.as_retriever()
qa=RetrievalQA.from_llm(llm=llm,prompt=prompt,retriever=retriever)
qa.run("What is Task Decomposition?")