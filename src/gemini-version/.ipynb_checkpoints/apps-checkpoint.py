'''
Chat With Multiple PDF Documents With Langchain And Google Gemini Pro #genai #googlegemini


Require VPN for Hong kong runing the Generative AI

go to makesuite generte  API key
https://makersuite.google.com/


set your google API key to environment variable 

'''

import streamlit as st  # import stremlit
from  PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter # text splitter
import os


from langchain_google_genai import GoogleGenerativeAIEmbeddings # google gemini
import google.generativeai as genai # google gemini
# from langchain.vectorstores import FAISS # vector store use FAISS # oold version
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI # google gemini
from langchain.chains.question_answering import load_qa_chain # question answering chart
from langchain.chains import RetrievalQAChain # retrieval question answering chart
from langchain.prompts import PromptTemplate # prompt template
# import langchain
from dotenv import load_dotenv # load environment variable


#load environment variable
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # set key to environment variable

# Set up the model
generation_config = {
  "temperature": 0.55,#0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}


safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]


#get pdf text
def getPDFText(pdf_docs):
    '''
    get pdf text from pdf docs
    '''
    text="" 
    for pdf in pdf_docs: #loop through pdf docs
        pdf_reader= PdfReader(pdf) #read pdf file
        for page in pdf_reader.pages: # loop through pdf pages 
            text+= page.extract_text() # extract text from page and add to text variable
    return  text # return text variable


#get text chunks (split text into chunks)
def getTextChunks(text, chunk_size=10000, chunk_overlap=1000):
    #inital Text splitter function 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap) # split text into chunks
    chunks = text_splitter.split_text(text) # split text into chunks
    return chunks # return chunks


#get vector store
def getVectorStore(text_chunks ,model="models/embedding-001"):
    embeddings = GoogleGenerativeAIEmbeddings(model = model) # use google gemini for embedding
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) # use FAISS for vector store with google gemini embedding
    vector_store.save_local("faiss_index") # save vector store
    return vector_store # return vector store


#get conversational chain
def getConversationalChain(temperature=0.3, modelName="gemini-pro"):
    #promptTemplate
    promptTemplate = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model=ChatGoogleGenerativeAI(model=modelName, 
                                 temperature=temperature) # use google gemini model for conversational chain
    
    prompt = PromptTemplate(template = promptTemplate, 
                            input_variables = ["context", "question"]) # use prompt template for conversational chain
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) # load question answering chart

    return chain


# user input encode into query vector
def user_input(user_question, model= "models/embedding-001"):
    embeddings = GoogleGenerativeAIEmbeddings(model = model)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # load vector store
    docs = new_db.similarity_search(user_question)  # search for similar text in vector store

    chain = getConversationalChain() # get conversational chain

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)  # get response from conversational chain 
    
    print(response)
    st.write("Reply: ", response["output_text"])


# main function
def main():
    st.set_page_config("Chat with Multiple PDF") # set page config
    st.header("Chat with Multiple PDF using Gemini💁") #set header 

    user_question = st.text_input("Ask a Question from PDF Files") # get user question

    if user_question:
        user_input(user_question) # get user input


    # sidebar
    with st.sidebar:
        st.title("Menu:") # set sidebar title
        pdfDocs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True) # upload pdf files
        # submit and process button
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): 
                raw_text = getPDFText(pdfDocs) #
                text_chunks = getTextChunks(raw_text)
                getVectorStore(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main() # run main function