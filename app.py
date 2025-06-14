import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
# from langchain.chains import conversational_retrieval
# from langchain.llms import HuggingFacePipeline
# from langchain_community.llms import openai, huggingface_hub, huggingface_pipeline
from htmlTemplate import user_template, bot_template, css
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain_community.llms import HuggingFaceHub
from datetime import datetime


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

# def get_vectorstore(chunks):
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
#     return vector_store
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def get_conversation(vector_store):
    # model_id = "google/flan-t5-base"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    # pipe = pipeline("text2text-generation", 
    #                 model=model, 
    #                 tokenizer=tokenizer, 
    #                 max_new_tokens=256,
    #                 truncation=True)

    model_id = "google/flan-t5-large"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,          # increase if needed
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain
    
    #llm = HuggingFaceEndpoint(repo_id="google/flan-t5-small", 
    #                          temperature=0.5)
    
    # llm = HuggingFaceEndpoint(
    #     repo_id="google/flan-t5-small",
    #     task="text2text-generation",
    #     temperature=0.5,
    #     max_new_tokens=512
    # )

    # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vector_store.as_retriever(),
    #     memory=memory
    # )
    #return conversation_chain

def handle_user_input(input):
    now = datetime.now().strftime("%I:%M %p")
    response = st.session_state.conversation.invoke({'question': input})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content).replace("{{TIME}}", now), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content).replace("{{TIME}}", now), unsafe_allow_html=True)

def main():
    load_dotenv()
    #os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
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
            with st.spinner("Processing..."):

                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #get the text chunks 
                chunks = get_chunks(raw_text)
                #st.write(chunks)
                
                #create a vector store
                vector_store = get_vectorstore(chunks)
                #st.write(vector_store)
                st.write("Number of vectors:", len(vector_store.index.reconstruct_n(0, vector_store.index.ntotal)))

                #creating a conversation chain
                st.session_state.conversation = get_conversation(vector_store)

                st.success("Processing complete! You can now start asking questions about your documents.")

if __name__ == '__main__':
    main()
