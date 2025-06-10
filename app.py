import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

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


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

    st.header("Chat with Multiple PDFs :books:")
    st.text_input("Ask a question about your documents: ")

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
                st.write(chunks)
                break
                #create a vector store

if __name__ == '__main__':
    main()