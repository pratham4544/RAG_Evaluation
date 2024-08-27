import streamlit as st
import pandas as pd
from src.helper import ResponseLLM

def create_download_button(dataframe, filename):
    csv = dataframe.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name=filename, mime='text/csv')

def process_uploaded_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Check if required columns are present
    if 'questions' in df.columns and 'ground_truths' in df.columns:
        st.success("CSV file loaded successfully!")
        st.write("Here is a preview of your CSV file:")
        st.write(df.head())
        
        questions_list = df['questions'].tolist()
        ground_truths_list = df['ground_truths'].tolist()
        
        return questions_list, ground_truths_list
    else:
        st.error("CSV file must contain 'questions' and 'ground_truths' columns.")
        return None, None

def main():
    st.header('RAG Evaluations')
    
    # File uploader to allow users to upload a CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    pdf_file = st.file_uploader("Upload your reference PDF file", type=['pdf'])
    
    if st.button('Submit'):
        if uploaded_file is not None:
            if pdf_file is not None:
                questions_list, ground_truths_list = process_uploaded_file(uploaded_file)
                
                if questions_list and ground_truths_list:
                    st.write('Started Working..')

                    # Save the uploaded PDF temporarily
                    with open(pdf_file.name, mode='wb') as f:
                        f.write(pdf_file.getvalue())

                    response = ResponseLLM()
                    chain, retriever = response.llm_response(pdf_file.name)

                    st.write('Generating Answer..')
                    model_answer, model_contexts = response.store_response(questions_list, chain, retriever)

                    st.write('RAGAS Evaluation Starts..')
                    ragas_result = response.ragas_eval(questions_list, ground_truths_list, model_answer, model_contexts)
                    st.dataframe(ragas_result.head())
                    create_download_button(ragas_result, "ragas_results.csv")

                    # st.write('BERT Evaluation Starts..')
                    # bert_score = response.bert_eval(questions_list, model_answer, ground_truths_list)
                    # st.dataframe(bert_score.head())

                    # st.write('Phoenix Evaluation Starts..')
                    # phoenix_result = response.phoenix_eval(questions_list, model_answer, model_contexts)
                    # st.dataframe(phoenix_result.head())

                    # st.write('Vectra Evaluation Starts..')
                    # vectra_result = response.vectra_eval(questions_list, model_contexts, model_answer, ground_truths_list)
                    # st.dataframe(vectra_result.head())
            else:
                st.error("Please upload a PDF file.")
        else:
            st.info("Please upload a CSV file.")

if __name__ == "__main__":
    main()
