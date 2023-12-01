import bs4
from langchain.llms import LlamaCpp
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=4096,
#    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#    verbose=True,# Verbose is required to pass to the callback manager
)
# 
prompt = PromptTemplate(
    input_variables=["question","chat_history"],
    template="""[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
Chat_history:{chat_history}
Question: {question} 
 
Answer: [/INST]
""",
)
loader = loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(docs)

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
underlying_embeddings = LlamaCppEmbeddings(model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf")
fs = LocalFileStore("/home/dubenhao/cache_cr/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, fs)

vectorstore = Chroma.from_documents(splits,cached_embedder)
retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever,condense_question_prompt=prompt, memory=memory)

questions=[
    "How do agents use Task decomposition?",
    "What are the various ways to implement memory to support it?"
]
for question in questions:
    result = qa(question)
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")