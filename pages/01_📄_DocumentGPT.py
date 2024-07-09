import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄"
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)
llm_for_memory = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
    ]
)

@st.cache_resource
def get_memory():
    return ConversationSummaryBufferMemory(
        llm=llm_for_memory,
        max_token_limit=120,
        return_messages=True
    )

memory = get_memory()

@st.cache_data(show_spinner="Embedding file..")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(file_path)

    spliter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=spliter)
    embeddings = OpenAIEmbeddings()

    cache_dir = LocalFileStore(f'./.cache/embeddings/{file.name}')
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings, document_embedding_cache=cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cache_embeddings) # Chroma / FAISS
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def load_memory(input):
    return memory.load_memory_variables({})["history"]

def clear_memory():
    memory.clear()

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
        Context: {context}
        """
     ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
st.title("DocumentGPT")

st.markdown("""
    Welcome!
    
    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on the sidebar.
""")


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
        on_change=clear_memory)

if file:
    retriever = embed_file(file)
    message = st.chat_input("")

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file.")

    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "history": RunnableLambda(func=load_memory),
            "question": RunnablePassthrough()
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
        memory.save_context(
            {"input": message}, {"output": response.content},
        )
        # send_message(response.content, "ai")
        # docs = retriever.invoke(message)
        # docs = "\n\n".join(document.page_content for document in docs)
        # prompt = template.format_messages(context=docs, question=message)
        # llm.predict_message(prompt)
else:
    st.session_state["messages"] = []
    memory = get_memory()

