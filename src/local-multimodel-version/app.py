'''
Chat With Multiple PDF Documents With Langchain And 

Open LLM 

'''



import streamlit as st  # import stremlit
from  PyPDF2 import PdfReader
import os, json, time, gc


import numpy as np
import pandas as pd
import transformers
import torch
from transformers import (AutoTokenizer, 
                          BitsAndBytesConfig,
                         AutoModelForCausalLM,
                         TrainingArguments)


from langchain_community.document_loaders import (TextLoader,
                                                  PyMuPDFLoader,
                                                  PyPDFDirectoryLoader,
                                                  PyPDFLoader)

from pypdf import PdfReader

from langchain_text_splitters import (RecursiveCharacterTextSplitter,
                                      CharacterTextSplitter ,
                                       SentenceTransformersTokenTextSplitter)   

from langchain.prompts.prompt import  PromptTemplate

from langchain_community.vectorstores import FAISS #, Chroma,  Pinecone # old version of VectorStore


from langchain_google_genai import ChatGoogleGenerativeAI # google gemini
from langchain.chains.question_answering import load_qa_chain # question answering chart
# from langchain.chains import RetrievalQAChain # retrieval question answering chart

# from langchain.embeddings import HuggingFaceEmbeddings # huggingfaceEmbedding deprecated , please use sentencetransformers 
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import Dataset, DatasetDict, load_dataset


# import langchain
from dotenv import load_dotenv # load environment variable



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)
if device.type == "cuda":
    model_kwargs = {"device": "cuda"}
    multiProcess=  False#True # 
else:
    model_kwargs = {"device": "cpu"}
    multiProcess= False


#load environment variable
load_dotenv()



class CFG:
    OFFLINE = False #True # for Test offline environment
    USE_LLAMA3 = False # for GPU version
    USE_GAMMA2B = True # for GPU version
    USE_GAMMA7B = False 
    USE_GEMMA2_9B = False # for GPU version only 
    TASK_GEN = True # for generative Text output task (suitable for RAG project)
    TEST_LLM = True
    USE_LMSTUIDO = False # for local LLM modle 
    model1 = "meta-llama/Meta-Llama-3-8B-Instruct"  # llama3 8B
    model2 =  "google/gemma-1.1-2b-it" #  gemma 2B
    model3 = "google/gemma-7b-it"# gemma 7B
    model4 =  "google/gemma-2-9b-it" # gemma 2 9B
    model5 =  'yentinglin/Llama-3-Taiwan-8B-Instruct' # Chinese version of llama3
    model6 = 'Qwen/Qwen-7B' # support Chinese version 
    model7 = "THUDM/chatglm-6b" # support Chinese version
    embedModel1 = 'intfloat/multilingual-e5-small' # for embedding model support chinese
    embedModel2 = "all-MiniLM-L6-v2"
    embedModel3 = "BAAI/bge-base-en-v1.5" # for embedding model support chinese
    embedModel4 = "BAAI/bge-m3" # for multilingual embedding model
    FEW_SHOT_TEST= False#True
    USE_RAG = True#False#False #True#True , in this project, prefer use fine tuning for p
    USE_WANDB = True#True # for  LLM evalution and debug , track fine tuning performance
    USE_TRULENS = False # for LLM evalution For RAG prefer 
    USE_DEEPEVAL = False # for LLM evalution   (require openAI API key)
    USE_TRAIN =  False #True #False#True Much be use GPU for Training 
    loggingSteps= 10#100 #100, #20, #5,#10,
    USE_FAISS = False#True # For RAG VectorDB
    USE_CHROMA = False#True #False # for RAG VectorDF
    USE_PINECONE = True#False #True # for RAG VectorDF
    maxTrainData = 200#3500#5000 #10000#5000 #10000
    maxEvalData = 20#100 # 20 
    maxToken=  512#768#512#768 # 512 for test only



# Template Prompt for conversational chain
templatePrompt1 = """Question: {question}.\nOnly require given final result in JSON format with key 'answer'
            """
templatePrompt2 = "Answer the user Question.\n###\n{format_instructions}\n###\nQuestion: {query}\n"


huggingfaceToken = os.getenv("HuggingFace") #get huggeface token from .env file

def embeddingModelInit(modelName):
    '''
    initialize embedding model
    '''
    embeddings = HuggingFaceEmbeddings(model=modelName, model_kwargs= model_kwargs, multi_process=multiProcess)
    return embeddings


def LLMInit():
    '''
    initialize LLM model
    '''
    # Quantized Config for GPU support only
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True # Activate nested quantization for 4-bit base models (double quantization)

    )
    if device.type == "cuda": # use 7b/8b/9b model gain performance
        if CFG.USE_LLAMA3:
            modelSel = CFG.model1
            llmModel = "llama3_8b"
        
        elif CFG.USE_GEMMA2_9B:
            modelSel = CFG.model4
            llmModel = "gemma2_9b"
        
        elif CFG.USE_GAMMA2B:
            modelSel = CFG.model2
            llmModel = "gemma_2b"

        elif CFG.USE_GAMMA7B:
            modelSel = CFG.model3
            llmModel = "gemma_7b"
        else:
            modelSel = CFG.model2
            llmModel = 'gemma_2b'
            
        model = AutoModelForCausalLM.from_pretrained(modelSel,  device_map="auto",
                                                    quantization_config= bnb_config ,
                                                    token=huggingfaceToken)
        tokenizer = AutoTokenizer.from_pretrained(modelSel, token=huggingfaceToken)
        tokenizer.padding_side = "right"
    else:
        model = AutoModelForCausalLM.from_pretrained(modelSel, device_map="auto", 
                                                     token=huggingfaceToken)
        tokenizer = AutoTokenizer.from_pretrained(modelSel, token=huggingfaceToken) # inital tokenizer
        tokenizer.padding_side = "right"
    print("Selected Model : ", llmModel) 
    print(model)# print model structure

    return model, tokenizer

def generateResponse(query,  model, tokenizer,  maxOutToken = 512):
    """
    Direct send message to LLM model, get response
    """
    inputIds = tokenizer(query, return_tensors="pt").to(device)
    response = model.generate(**inputIds,
                              do_sample=True,
                              top_p=0.95,
                              top_k = 3,
                              temperature=0.5,
                              max_new_tokens= maxOutToken,
                             )
    return tokenizer.decode(response[0][len(inputIds["input_ids"]):], skip_special_tokens = True)

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
def getTextChunks(text, chunk_size=800, chunk_overlap=50):
    #inital Text splitter function 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap) # split text into chunks
    chunks = text_splitter.split_text(text) # split text into chunks
    return chunks # return chunks


#get vector store
def getVectorStore(text_chunks ,model=CFG.embedModel3):
    embeddings = embeddingModelInit(model)
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
    #model
    model=ChatGoogleGenerativeAI(model=modelName, 
                                 temperature=temperature) # use google gemini model for conversational chain
    
    prompt = PromptTemplate(template = promptTemplate, 
                            input_variables = ["context", "question"]) # use prompt template for conversational chain
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) # load question answering chart

    return chain


# user input encode into query vector
def user_input(user_question, model= CFG.embedModel3):
    # embeddings = GoogleGenerativeAIEmbeddings(model = model)
   
    # loading Vector DB stored document 
    embeddings = embeddingModelInit(model)
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
    model , tokenizer = LLMInit() # initial LLM model 
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