from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,# Verbose is required to pass to the callback manager
)
# Prompt
prompt = PromptTemplate(
    input_variables=["question","chat_history"],
    template="""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>
Here is the chat history:{chat_history}
Answer the following question in one sentence:{question} [/INST]
""",
)
#memory
memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)
conversation(
    {"question": "Translate this sentence from English to French: I love programming."}
)
conversation({"question": "Now translate the sentence to German."})
conversation({"question": "Now translate the sentence to Japanese."})
conversation({"question": "Now translate the sentence to Chinese."})


