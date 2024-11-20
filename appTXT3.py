#Se agrega modo chat y se traduce a español
#se modifica nombre de la pestaña
#Se cambia modelo a gpt4o

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Inicializar embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Función para leer archivos PDF
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Función para dividir el texto en chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Crear y guardar el vector store
def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

# Obtener la cadena de conversación
def get_conversational_chain(tools, ques):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key="")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in
                the provided context, say "answer is not available in the context" and don't provide the wrong answer.""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": ques})
    
    return response['output']

# Función para manejar la entrada del usuario
def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answers to queries from the pdf")
    
    return get_conversational_chain(retrieval_chain, user_question)

# Función principal
def main():
    st.set_page_config("Chat with PDF3") #nombre de la pestaña
    st.header("Chat with PDF files")

    # Inicializar el estado de los mensajes si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de la pregunta del usuario
    if user_question := st.chat_input("Escribe tu pregunta sobre los archivos PDF:"):
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Mostrar el mensaje del usuario en el chat
        with st.chat_message("user"):
            st.markdown(user_question)

        # Procesar la pregunta
        with st.spinner("Procesando la consulta..."):
            respuesta = user_input(user_question)
            
            # Mostrar la respuesta de la IA
            with st.chat_message("assistant"):
                st.markdown(respuesta)

            # Guardar la respuesta del asistente en el estado de la sesión
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

    # Subida de PDF en el sidebar
    with st.sidebar:
        st.title("Archivos:")
        logo = "logo.png"  # Cargar logo
        st.sidebar.image(logo, width=150)  # Mostrar logo
        pdf_doc = st.file_uploader("Sube tus archivos PDF y haz clic en 'Procesar'", accept_multiple_files=True)
        
        if st.button("Procesar"):
            with st.spinner("Procesando archivos..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Procesamiento completado")

if __name__ == "__main__":
    main()
