__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.llms import Ollama 
import os
from pathlib import Path
from pypdf import PdfReader
from langchain.docstore.document import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community import embeddings 
from langchain_community.vectorstores import Chroma 
from langchain_core.runnables import RunnablePassthrough

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


llm = Ollama(model = "llama3.2")



def get_document_text(uploaded_file, title=None):
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)

    pdf_reader = PdfReader(uploaded_file)
    for num, page in enumerate(pdf_reader.pages):
        page = page.extract_text()
        doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
        docs.append(doc)



    return docs

def load_pdf_files(data_dir="data"):
    docs = []
    paths = Path(data_dir).glob('**/*.pdf')
    for path in paths:
        print(path)
        this_lst = get_document_text(path, title=None)
        docs += this_lst
    return docs



@st.cache_resource
def embeddings_():
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True 
    )
    docs = load_pdf_files(data_dir="data")
    all_splits = text_splitter.split_documents(docs)


    embedding = embeddings.OllamaEmbeddings(
        model="mxbai-embed-large" 
    )

    vectorstore = Chroma(
    collection_name = 'rag_chroma',
    embedding_function = embedding,
    persist_directory=os.path.join("store/", 'rag_chroma'))

    vectorstore.add_documents(all_splits)
    vectorstore.persist()

    return vectorstore




retriever = embeddings_()

retriever = retriever.as_retriever()



#Create a chain for chat history


qa_system_prompt =  """You are an assistant for question-answering tasks in the medical domain. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know, do not invent, or ask the user to 
reformulate his question. \


{context}"""


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return StreamlitChatMessageHistory(key="messages")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return get_session_history # contextualize_q_chain  #
    else:
        return input["question"]
    

# Function to clear chat history
def clear_chat_history():
    
    
    if "messages" in st.session_state:
        del st.session_state["messages"]

    st.session_state.messages = [{"role": "assistant", "content": "Hello, I am Uriel your medical doctor, How can I help you?"}]
    


def chat(question):

    chat_history = []
    rag_chain = (
        RunnablePassthrough.assign(
            context = contextualized_question | retriever | format_docs
        )
        | qa_prompt 
        | llm
    )

    ## Extract and print the context
    ## Uncomment it if you want to see the context used by your LLM
    #context = (contextualized_question | retriever | format_docs).invoke(
    #    {
    #        "question": question,
    #        "chat_history": chat_history
    #    }
    #    )
    ## Uncomment the line below it if you want to see the context used by your LLM
    #st.markdown(context)

    response = rag_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history
        },
        config={"configurable": {"session_id": "foo"}}
    )

    return response





def show_ui(prompt_to_user="Hello, I am Uriel your medical doctor, How can I help you?"):

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]
    with st.sidebar:
        st.button("Delete conversations history", on_click=clear_chat_history)
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User-provided prompt
    if prompt := st.chat_input(placeholder="Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(prompt) 
                st.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


def run():
    ready = True


    if ready:
        show_ui("Hello, I am Uriel your medical doctor, How can I help you?")
    else:
        st.stop()


run()