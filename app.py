import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceInstructEmbeddings
#from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
from langchain_community.llms import openai, huggingface_hub, huggingface_pipeline
#from langchain.llms import openai, huggingface_hub
from htmlTemplate import user_template, bot_template, css
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def get_conversation(vector_store):
    #llm = openai()
    #llm = huggingface_hub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # llm = huggingface_hub(
    # repo_id="google/flan-t5-xxl",
    # model_kwargs={
    #     "temperature": 0.5,
    #     "max_length": 512
    #     }
    # )

    model_id = "google/flan-t5-base"  
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = huggingface_pipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = conversational_retrieval.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(input):
    response = st.session_state.conversation({'question': input})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")
    input = st.text_input("Ask a question about your documents: ")
    if input:
        handle_user_input(input)


    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your documents here and click on Process Button", accept_multiple_files=True)

        if st.button("Process"):
            while st.spinner("Processing..."):

                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #get the text chunks 
                chunks = get_chunks(raw_text)
                #st.write(chunks)
                
                #create a vector store
                vector_store = get_vectorstore(chunks)

                #creating a conversation chain
                st.session_state.conversation = get_conversation(vector_store)

if __name__ == '__main__':
    main()