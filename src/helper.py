from langchain import FAISS, PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ai21 import AI21Embeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import time
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification
import os
from dotenv import load_dotenv
from src.prompt import *


load_dotenv()

class ResponseLLM:
    def __init__(self, model=None, embeddings=None, vectra_model=None):
        # Initialize the model and embeddings, using defaults if none provided
        self.model = model or ChatGroq(temperature=0.7, model="llama3-70b-8192", api_key=os.environ['GORQ_API_KEY'])
        self.embeddings = embeddings or GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectra_model = vectra_model or AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True)

def load_pdf(ebook):
    # Prepare vector store (FAISS)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, add_start_index=True)
    loader = PyPDFLoader(ebook)
    documents = loader.load_and_split(text_splitter)
    db = FAISS.from_documents(documents, AI21Embeddings())
    return db

def load_llm(db):
    # Create LLM chain
    llm = ChatGroq(temperature=0.0, model="llama3-70b-8192")
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
    retriever = db.as_retriever(search_kwargs={"k": 3})
    harry_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever, prompt=prompt)
    return harry_qa_chain, retriever

def get_answer(question, harry_qa_chain, retriever):
    answers = []
    contexts = []
    time.sleep(1)
    for q in question:
        result = harry_qa_chain.invoke({"query": q})
        answers.append(result['result'])
        contexts.append([doc.page_content for doc in retriever.get_relevant_documents(q)])
    return answers, contexts

def load_data(question, answer, context):
    data = {
        "question": question,
        "answer": answer,
        "contexts": context,
    }
    df = pd.DataFrame(data)
    df['contexts'] = df['contexts'].apply(lambda x: ' '.join(x))
    return df

def check(pairs):
    return ['Factual' if score >= 0.5 else 'Hallucinated' for score in pairs]

def vectra_classifications(df, vectra_model):
    pairs = list(zip(df['contexts'], df['answer']))
    scores = vectra_model(pairs)  # Replace this with the actual method call to get predictions
    final = check(scores)
    return final, scores

def store_vectra(questions, answer, final, score):
    vectra_score = pd.DataFrame({
        "question": questions,
        "answer": answer,
        "vectra": final,
        "score": np.round(score, 2)
    })
    vectra_score.to_csv('vectra_predict.csv', index=False)
    return vectra_score
