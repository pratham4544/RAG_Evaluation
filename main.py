# streamlit_app.py

import streamlit as st
from src.helper import *
from src.data_loader import *

# ResponseLLM = ResponseLLM()
     
def main():
    st.title("RAG Evaluation App")
    
    csv_file = st.file_uploader("Upload CSV File", type=["csv"])
    document_file = st.file_uploader("Upload Document File", type=["txt", "pdf", "docx"])
    
    if csv_file and document_file:
        csv_data = load_csv(csv_file)
        document_data = load_document(document_file)
        
        db = load_pdf(document_data)
        harry_qa_chain, retriever = load_llm(db)
        
        question = csv_data['question'].tolist()
        model_answer, model_context = get_answer(question,harry_qa_chain,retriever)
        
        data = load_data(question, model_answer, model_context)
        
        final, score = vectra_classifications(data, ResponseLLM.vectra_model)
        
        vectra_output =store_vectra(questions=question,answer=model_answer, final=final,score=score)

        st.subheader("Detailed Evaluation Data")
        st.dataframe(vectra_output)
    
if __name__ == "__main__":
    main()
