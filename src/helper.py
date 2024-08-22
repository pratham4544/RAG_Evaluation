from langchain import FAISS, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from datasets import Dataset
from ragas.metrics import ( FaithulnesswithHHEM,
                           faithfulness,
                            answer_relevancy,
                            context_utilization,
                            context_precision,
                            context_utilization,
                            answer_correctness,
                            context_entity_recall,
                            answer_similarity )
from ragas import evaluate
from langchain_groq import ChatGroq
from ragas.run_config import RunConfig
from langchain_ai21 import ChatAI21
from langchain import FAISS, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_ai21 import ChatAI21
from bert_score import score
from bert_score import BERTScorer
import pandas as pd
from langchain import FAISS, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from phoenix.evals import (
    HALLUCINATION_PROMPT_RAILS_MAP,
    HALLUCINATION_PROMPT_TEMPLATE,
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)
from langchain_groq import ChatGroq
from langchain_ai21 import ChatAI21
import os
import pandas as pd
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
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification
from langchain_community.document_loaders import PyPDFDirectoryLoader



load_dotenv()


def check(pairs):
  final = []
  for i in pairs:
    if i>=0.5:
      final.append('Factual')
    else:
      final.append('Hallucinated')
  return final


class ResponseLLM:
    def __init__(self, model=None, llm = None, embeedings=None, critic_model=None, openai_model=None):
        self.model = model  or ChatGroq(temperature=0, model="llama3-70b-8192")
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        self.embeedings = embeedings or GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.critic_model = ChatGroq(temperature=0, model_name='llama-3.1-70b-versatile')
        self.openai_model = OpenAIModel(model="gpt-3.5-turbo",temperature=0.0)

    def llm_response(pdf_path):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)

        db = FAISS.from_documents(texts, self.embeedings)

        retriever = db.as_retriever()

        PROMPT_TEMPLATE = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question:
        {question}

        Your answer:
        """

        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


        chain = RetrievalQA.from_llm(llm=self.llm, retriever=retriever, prompt=prompt)

        return chain, retriever
    
    def store_response(questions, chain, retriever):
        model_answer = []
        model_contexts = []

        for question in questions:
            result = chain.invoke(question)
            model_answer.append(result['result'])
            model_contexts.append([docs.page_content for docs in retriever.get_relevant_documents(question)])

        return model_answer, model_contexts


    def ragas_eval(questions, ground_truths, model_answer, model_contexts):

        ragas_dataset = {
            'question': questions,
            'ground_truth': ground_truths,
            'answer': model_answer,
            'contexts': model_contexts
            }

        dataset = Dataset.from_dict(ragas_dataset)

        faithfullness = FaithulnesswithHHEM()

        results = evaluate(dataset,
                        metrics=[faithfullness,
                                faithfulness,
                                    answer_relevancy,
                                    context_utilization,
                                    context_precision,
                                    context_utilization,
                                    answer_correctness,
                                    context_entity_recall,
                                    answer_similarity],
                        llm= self.critic_llm, embeddings= self.embeddings)

        results.to_pandas().to_csv('ragas_results.csv')
        
        return results


    def bert_eval(questions, model_answer, ground_truths):
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score(model_answer, ground_truths)
        bert_score = pd.DataFrame({'questions': questions, "Precision": P, 'Recall':R, 'F1 Score':F1})
        bert_score.to_csv('BERT_Scores.csv')

        return bert_score
    
    def phoenix_eval(questions, model_answer, model_contexts):

        df = pd.DataFrame({'input': questions, 'output': model_answer, 'reference': model_contexts})

        rails = list(HALLUCINATION_PROMPT_RAILS_MAP.values())

        hallucination_classifications = llm_classify(
            dataframe=df,
            template=HALLUCINATION_PROMPT_TEMPLATE,
            model=self.openai_model,
            rails=rails,
            provide_explanation=True,
        )


        hallucination_classifications.to_csv('hallucination_classifications.csv')

        return hallucination_classifications
    
    
    def vectra_eval(questions,model_contexts, model_answer, ground_truths):

        model = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True)

        pairs = list(zip(model_contexts, model_answer))
        score = model.predict(pairs)
        final = check(score)

        vectra_score = pd.DataFrame({
            "questions": questions,
            'answer': model_answer,
            'context': model_contexts,
            'ground_truths':ground_truths,
            "vectra": final,
            'score': np.round(score, 2)

        })

        vectra_score.to_csv('vectra_predict.csv')

        return vectra_score


