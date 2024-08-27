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
    
    # API Key input
    # openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    # Checkboxes for selecting evaluations
    run_ragas = st.checkbox('Run RAGAS Evaluation')
    run_bert = st.checkbox('Run BERT Evaluation')
    run_phoenix = st.checkbox('Run Phoenix Evaluation')
    run_vectra = st.checkbox('Run Vectra Evaluation')
    
    # File uploader to allow users to upload a CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    pdf_file = st.file_uploader("Upload your reference PDF file", type=['pdf'])
    
    if st.button('Submit'):
        if uploaded_file is not None:
            if pdf_file is not None:
                if openai_api_key:
                    questions_list, ground_truths_list = process_uploaded_file(uploaded_file)
                
                    if questions_list and ground_truths_list:
                        st.write('Started Working..')

                        # Save the uploaded PDF temporarily
                        with open(pdf_file.name, mode='wb') as f:
                            f.write(pdf_file.getvalue())

                        response = ResponseLLM()

                        # Update the OpenAI model with the provided API key
                        # response.openai_model = OpenAIModel(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=openai_api_key)
                        
                        chain, retriever = response.llm_response(pdf_file.name)

                        st.write('Generating Answer..')
                        model_answer, model_contexts = response.store_response(questions_list, chain, retriever)

                        # RAGAS Evaluation
                        if run_ragas:
                            st.write('RAGAS Evaluation Starts..')
                            ragas_result = response.ragas_eval(questions_list, ground_truths_list, model_answer, model_contexts)
                            st.dataframe(ragas_result.head())
                            create_download_button(ragas_result, "ragas_results.csv")

                        # BERT Evaluation
                        if run_bert:
                            st.write('BERT Evaluation Starts..')
                            bert_score = response.bert_eval(questions_list, model_answer, ground_truths_list)
                            st.dataframe(bert_score.head())
                            create_download_button(bert_score, "bert_score_results.csv")

                        # Phoenix Evaluation
                        if run_phoenix:
                            st.write('Phoenix Evaluation Starts..')
                            phoenix_result = response.phoenix_eval(questions_list, model_answer, model_contexts)
                            st.dataframe(phoenix_result.head())
                            create_download_button(phoenix_result, "phoenix_results.csv")

                        # Vectra Evaluation
                        if run_vectra:
                            st.write('Vectra Evaluation Starts..')
                            vectra_result = response.vectra_eval(questions_list, model_contexts, model_answer, ground_truths_list)
                            st.dataframe(vectra_result.head())
                            create_download_button(vectra_result, "vectra_results.csv")
                    else:
                        st.error("Please check the content of your CSV file.")
                else:
                    st.error("Please provide your OpenAI API key.")
            else:
                st.error("Please upload a PDF file.")
        else:
            st.info("Please upload a CSV file.")

if __name__ == "__main__":
    main()
