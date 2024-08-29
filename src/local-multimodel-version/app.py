'''
Chat With Multiple PDF Documents With Langchain And 

Open LLM 

'''



import streamlit as st  # import stremlit
from  PyPDF2 import PdfReader
import pymupdf
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
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings # new version of huggingface embedding
from sentence_transformers import SentenceTransformer
from datasets import Dataset, DatasetDict, load_dataset


# import langchain
from dotenv import load_dotenv # load environment variable



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)



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

promptTemplate3 = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

templatePrompt4 = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
CONTEXT: {context}
Provide answer and rethinking multiple step by step from Question: {question}
"""
templatePrompt5 = """
you are act as Mathematician, solve the math problem reasonable and logical from given question follow the requirement as below:
CONTEXT: {context}
Provide answer and rethinking multiple step by step from Question: {question}
Only Output answer in json format with key "answer" and "explanation" 
"""

huggingfaceToken = os.getenv("HuggingFace") #get huggeface token from .env file

# print("HuggingFace Token: ", huggingfaceToken)

def embeddingModelInit(modelName,  gpu=False):
    '''
    initialize embedding model
    '''
    global device
    if device.type == "cuda":
        if gpu:
            model_kwargs = {"device": "cuda"}
            multiProcess=  False#True # 
        else:
            model_kwargs = {"device": "cpu"}
            multiProcess= False
    else:
        model_kwargs = {"device": "cpu"}
        multiProcess= False
    # embeddings = HuggingFaceEmbeddings(model=modelName, model_kwargs= model_kwargs, multi_process=multiProcess)
    # embeddings = SentenceTransformer(modelName)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=modelName,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
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

#Extract data from multiple pdf files
def getPDFText(pdf_docs):
    '''
    get pdf text from pdf docs by PyPDF2
    '''
    text="" 
    for pdf in pdf_docs: #loop through pdf docs
        print(pdf)
        pdf_reader= PdfReader(pdf) #read pdf file
        for page in pdf_reader.pages: # loop through pdf pages 
            text+= page.extract_text() # extract text from page and add to text variable
    return  text # return text variable


def extractTextFromPage(page):
    '''get text by pymupdf
    '''
    text = page.get_text()
    return text

def extractTableFromPage(page):
    '''
    extract table from page by pymupdf
    '''
    tabs = page.find_tables()
    print(f"{len(tabs.tables)} found on {page}") # display number of found tables
    for i, tab in enumerate(tabs.tables):
        print(f"Table {i+1} : {tab.extract()}")
    return tabs

def extractImageFromPage(page):
    image_list = page.get_images()
    imginfo = page.get_image_info()
    print(imginfo)
    print(image_list)
    return image_list

def getPDFData(pdf_docs):
    '''
    get pdf text from pdf docs by pymupdf
    '''
    text="" 
    for pdf in pdf_docs: #loop through pdf docs
        # print("PDF file : ", type(pdf))
        # print("PDF file : ", pdf)
        pdf_bytes = pdf.getvalue()  # Read the PDF file into bytes
        pdf_reader= pymupdf.open(stream=pdf_bytes, filetype="pdf") #read pdf file
        for page in pdf_reader:# loop through pdf pages
            print("Page: ", page)
            text += extractTextFromPage(page) 
            # text+= page.get_text() # extract text from page and add to text variable
    return  text # return text variable

#get text chunks (split text into chunks)
def getTextChunks(text, chunk_size=600, chunk_overlap=50):
    #inital Text splitter function 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap) # split text into chunks
    chunks = text_splitter.split_text(text) # split text into chunks
    print(f"Chunks Size: {len(chunks)}")
    return chunks # return chunks


#get vector store
def getVectorStore(text_chunks ,model=CFG.embedModel3):
    embeddings = embeddingModelInit(model)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) # use FAISS for vector store with google gemini embedding
    vector_store.save_local("faiss_index") # save vector store
    return vector_store # return vector store

def getEmbeddingEncode(query, embeddings):
    vector = embeddings.embed_query(query)
    print(len(vector))
    return vector


def userQuery(user_question, llmModel, tokenizer, embeddModel=CFG.embedModel3):
    # RAG retrieval model
    # loading Vector DB stored document 
    ragContext = "" # for reg context input to prompt
    embeddings = embeddingModelInit(embeddModel) # embedding model for vector store
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # load vector store
    num_docs = 3
    retriever  = new_db.as_retriever( 
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": num_docs})
    # docs = new_db.similarity_search(user_question)  # search for similar text in vector store
    docs = retriever.invoke(user_question) # get response from retriever
    # response = docs.copy()
    print("number of document RAG Response: ", len(docs))
    for doc in docs:
        ragContext += doc.page_content + "\n"
    #
    newPrompt = PromptTemplate(template = templatePrompt4, 
                            input_variables = ["context", "question"]) # use prompt template for conversational chain
    finalPrmpt = newPrompt.format(
            question=user_question,
            context=ragContext
        )
    response = generateResponse(finalPrmpt, llmModel, tokenizer, maxOutToken=512)
    st.write("Result : ", response)

# main function
def main():
    model , tokenizer = LLMInit() # initial LLM model 
    # Test LLM model
    # ret =generateResponse( query="What is Machine Learning?",   model=model,  tokenizer= tokenizer, maxOutToken=256)
    # print(ret) 
    # embeddings = embeddingModelInit(CFG.embedModel3)
    # vector = getEmbeddingEncode("Hello", embeddings)
    st.set_page_config("Chat with Multiple PDF") # set page config
    st.header("Chat with Multiple PDF using Open LLM") #set header 

    # LLM and RAG Retrieval Query
    user_question = st.text_input("Ask a Question from PDF Files") # get user question

    if user_question:
        # user_input(user_question) # get user input
        userQuery(user_question, llmModel=model, tokenizer=tokenizer) # get user query


    # sidebar
    with st.sidebar:
        st.title("Menu:") # set sidebar title
        pdfDocs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True, type=['pdf']) # upload pdf files
        # submit and process button
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Store PDF files docs in Vector Store
                if pdfDocs: 
                    # raw_text = getPDFText(pdfDocs) #
                    raw_text = getPDFData(pdfDocs)
                    print(f"Len of Raw Test: {len(raw_text)}")
                    text_chunks = getTextChunks(raw_text)
                    print("Text Chunks size : ", len(text_chunks))
                    getVectorStore(text_chunks)
                    st.success("Done")
                else:
                    print("No PDF Docs Found")


if __name__ == "__main__":
    main() # run main function