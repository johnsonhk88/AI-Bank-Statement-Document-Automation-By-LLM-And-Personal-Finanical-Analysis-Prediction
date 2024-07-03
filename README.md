# AI-Bank Statement document aution by LLM and Personal financial analysis


### Introdctions
#### Business/ Use case 
Every Month, we obtain a lot bank statement in pdf format document. We intend to calculate and summarize the personal expensive and income from bank statement, statistically analysis monthly and yearly income vs expense for personal finanial planing . It is taking time to handle and store data.

This project intended to be used LLM Model for the purpose of assisting the user obtain fully record from bank statement by RAG technique. Convert the bank statement from unstructured document format into structured format. Store the records into database. Use LLM nature language to query the bank statement and give us the report. 


### Technology use in this project
1. LLM Model 
- try to use different open LLM models (e.g. LLama3, gemma 2) , prefer use local open LLM models(planning inference LLM model at offline in local machine)

2. retrieval augmented generation (RAG) with langChain  
- using training dataset for improvement the Text summaration task for conference speakers
- using Advance RAG technique improve retrieval accuracy (e.g. re-ranking, query extensions, auto-merging)

3. Unstructure Document Preprocssing
- Because the input document complexity, include table, image (chart),  use like OCR, Vision transformer to extract the document content from bank statement. 

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

### Run Application
1. running steamlit run apps.py for develop the application
