# AI-Bank Statement document automatic by LLM and Personal financial analysis


### Introdctions
#### Business/ Use case 
Every Month, we obtain a lot bank statement in pdf format document. We intend to calculate and summarize the personal expensive and income from bank statement, statistically analysis monthly and yearly income vs expense for personal finanial planing . It is taking time to handle and store data.

This project intended to be used LLM Model for the purpose of assisting the user obtain fully record from bank statement by RAG technique. Convert the bank statement from unstructured document format into structured format. Store the records into database. Use LLM nature language to query the bank statement and give us the report. 


This project mainly can divide into three main parts: 
1. Data Extraction model for PDF file complex format 
2. Embedding model + Vector Database for Store PDF Retrival document
3. LLM Model + RAG technique from data retrieved from database with natural language queries by user 

### Technology use in this project
1. Unstructure Document Preprocssing
- Because the input document complexity, include table, image (chart), I will use several AI model  like OCR , commputer vision model, Vision transformer , layout transformer, Embedding model to extract and analysis the document content from bank statement.
- Complex layout/Context format Analysis by ML model 
- Level 1 analysis: Document layout Analysis
  - Use Computer vision (object detection) AI model to extract component in document content
    - Custom Train Object ddection Model (YOLO) for Detect/recogize the Document Layout Component
    - Detail of the Custom Train YOLO document layout detection model 
    - see my other github project (yolo-base-doc-layout-detection) :  <https://github.com/johnsonhk88/yolo-base-doc-layout-detection> 
  - then use different AI model analysis and extract different type of components context
  -
- Level 2 each component context 
  - use different AI model for extract and recognize different types of docunment components

- Level 3 High level task analysis
  - use AI model Entities 
  - use AI model Sentiment Analysis
  - use AI model Summarization 

- use advance rule base model or Machine learning  model :
  - group and reorganize the data into a user-friendly format. (no experience to build rule to graoup data)
  - Identify common denominators and create headers for each group. (no experence)
  - Display only the differences between similar items (e.g., window sizes, owners) as line items below each header. 
  - Automate the process using AI, enabling the system to self-learn and understand the data structure.  
  - Extract relevant data from PDFs with different layouts and formats.



2. retrieval augmented generation (RAG) with langChain  
- use Embedding model with VectorDB to Retrieve data values by query
- using training dataset for improvement the Text summaration task for conference speakers
- using Advance RAG technique improve retrieval accuracy (e.g. re-ranking, query extensions, auto-merging)

3. LLM Model / Multi-model 
- try to use different open LLM models / multi-model (e.g. LLama3, gemma 2) , prefer use local open LLM models(planning inference LLM model at offline in local machine)
- LLM model use for user-friendly documentation queries and retrieval information interface with natural language

4. LLM Model evaluation
- use truLens or W&B framework for evaluation and debug LLM performance
- LLM evaluation : Content relevance, Answer Relevance, accuary, recall, precision 

5. AI agent
- use AI agent automatically trigger multiple function 

6. VectorDB 
- use Vector DataBase to store the converted Document context into embedding vector
- use Vector Database can find document similarity 

7. SQL Database
- use to store the conference record
- use for query history conference record

8. FrontEnd UI
- first version will be used Streamlit for Frontend UI
- later versions will be Full stack with Backend Restful API



### Installation and Setup
1. use requirements.txt for installation package dependencies
2. you can setup virual environment by venv 
3. add your google api key to .env file  for enviroment variables
4. install pytesseract library for ubuntu linux , please run install-pytesseract-for-linux.sh script file 

### Run Application
1. For Development version:
    go to dev folder run jupyter notebook code for development new model/techique 
2. For Application GUI version: 
    running steamlit run apps.py for develop the application
