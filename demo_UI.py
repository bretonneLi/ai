from langchain.llms import LlamaCpp

from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate

from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

llm = LlamaCpp(
    model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,

)

template="""[INST]
    Use the following pieces ofcontext to answer the question.If no context provided, answer like a AI assistant. 
    
    {context} 
    Question: {question}
    Answer the question only in Chinese.[/INST]
"""


vectorstore=FAISS.load_local("/home/dubenhao/vectorstore/db_faiss", embeddings=LlamaCppEmbeddings(model_path="/home/dubenhao/llama/llama-2-7b-chat/ggml-model-q4_k_m.gguf"))
retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever,     
    chain_type_kwargs={
        "prompt":  PromptTemplate(
            template=template,
            input_variables=["context", "question"]),
    }
)



system_message = {"role": "system", "content": "You are a helpful assistant."}
#UI
from queue import Queue
from typing import Any
from langchain.schema import LLMResult
from anyio.from_thread import start_blocking_portal
job_done = object()

class MyStream(StreamingStdOutCallbackHandler):
    def __init__(self, q) -> None:
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.q.put(job_done)


import gradio as gr

with gr.Blocks() as demo:
    #Configure UI layout
    chatbot = gr.Chatbot(height = 600)
    with gr.Row():
        
        with gr.Column():
            #user input prompt text field
            user_prompt_message = gr.Textbox(placeholder="Please add user prompt here", label="User prompt")
            with gr.Row():
                clear = gr.Button("Clear Conversation", scale=2)
                submitBtn = gr.Button("Submit", scale=8)


    state = gr.State([])

    #handle user message
    def user(user_prompt_message, history):
        if user_prompt_message != "":
            return history + [[user_prompt_message, None]]
        else:
            return history + [["Invalid prompts - user prompt cannot be empty", None]]

    #chatbot logic for configuration, sending the prompts, rendering the streamed back genereations etc
    def bot(  user_prompt_message, history, messages_history):
        dialog = []
        bot_message = ""
        history[-1][1] = ""
           
        dialog = [
            {"role": "user", "content": user_prompt_message},
        ]
        messages_history += dialog
        
        #Queue for streamed character rendering
        q = Queue()

        #Update new llama hyperparameters
        
 

        #Async task for streamed chain results wired to callbacks we previously defined, so we don't block the UI
        async def task(prompt):
            ret = await qa_chain.run(prompt, callbacks=[MyStream(q)])
            return ret

        with start_blocking_portal() as portal:
            portal.start_task_soon(task, user_prompt_message)
            while True:
                next_token = q.get(True)
                if next_token is job_done:
                    messages_history += [{"role": "assistant", "content": bot_message}]
                    return history, messages_history
                bot_message += next_token
                history[-1][1] += next_token
                yield history, messages_history

    #init the chat history with default system message    
    def init_history(messages_history):
        messages_history = []
        messages_history += [system_message]
        return messages_history

    #clean up the user input text field
    def input_cleanup():
        return ""

    #when the user clicks Enter and the user message is submitted
    user_prompt_message.submit(
        user, 
        [user_prompt_message, chatbot], 
        [chatbot], 
        queue=False
    ).then(
        bot, 
        [  user_prompt_message, chatbot, state], 
        [chatbot, state]
    ).then(input_cleanup, 
        [], 
        [user_prompt_message], 
        queue=False
    )

    #when the user clicks the submit button
    submitBtn.click(
        user, 
        [user_prompt_message, chatbot], 
        [chatbot], 
        queue=False
    ).then(
        bot, 
        [ user_prompt_message, chatbot, state], 
        [chatbot, state]
    ).then(
        input_cleanup, 
        [], 
        [user_prompt_message], 
        queue=False
    )
    
    #when the user clicks the clear button
    clear.click(lambda: None, None, chatbot, queue=False).success(init_history, [state], [state])
    demo.queue().launch()