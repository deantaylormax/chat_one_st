
# %%
from dotenv import load_dotenv
import os
import dotenv
dotenv.load_dotenv()
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Define your username and password
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
USER1 = os.getenv("user1")
PASSWORD1 = os.getenv("password1")

# Initialize session state for logged_in and attempted_login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'attempted_login' not in st.session_state:
    st.session_state.attempted_login = False

if not st.session_state.get('logged_in', False):
    # User is not logged in, show login fields
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        st.session_state.attempted_login = True
        # Check against both sets of credentials
        if (username == USER1 and password == PASSWORD1) or (username == USERNAME and password == PASSWORD):
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
        else:  # This else belongs to the for loop, not the if statement
            if st.session_state.attempted_login:
                st.error("Incorrect username or password")
else:
    # User is logged in, show logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.attempted_login = False
        st.session_state.user_role = None
        st.rerun()  # Use experimental_rerun for the latest versions of Streamlit

def main():
    load_dotenv()
    # st.set_page_config(page_title="Talk to your PDF")
    st.header("Ask your PDF")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            st.write(response)
if st.session_state.logged_in:
    main()
# if __name__ == '__main__':
#     main()