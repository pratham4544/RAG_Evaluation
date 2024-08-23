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
                pdf_data = pdf_file.name
                st.write('PDF DATA ',pdf_data)
                
                if questions_list and ground_truths_list:
                    st.write('Started Working..')

                    response = ResponseLLM()

                    # with open(pdf_file.name, mode='wb') as w:
                    #     w.write(pdf_file.getvalue())

                    # text = response.get_text_file(pdf_data)

                    chain, retriever = response.llm_response(pdf_data)

                    st.write('Generating Answer..')
                    model_answer, model_contexts = response.store_response(questions_list, chain, retriever)

                    st.write('RAGAS Evaluation Starts..')
                    ragas_result = response.ragas_eval(questions_list, ground_truths_list, model_answer, model_contexts)
                    st.dataframe(ragas_result.head())
                    create_download_button(ragas_result, "ragas_results.csv")

                    st.write('BERT Evaluations Starts..')
                    bert_score = response.bert_eval(questions_list, model_answer, ground_truths_list)
                    st.dataframe(bert_score.head())
                    create_download_button(bert_score, "bert_score_results.csv")

                    st.write('Phoenix Evaluations Starts..')
                    phoenix_result = response.phoenix_eval(questions_list, model_answer, model_contexts)
                    st.dataframe(phoenix_result.head())
                    create_download_button(phoenix_result, "phoenix_results.csv")

                    st.write('Vectra Evaluations Starts..')
                    vectra_result = response.vectra_eval(questions_list, model_contexts, model_answer, ground_truths_list)
                    st.dataframe(vectra_result.head())
                    create_download_button(vectra_result, "vectra_results.csv")
            else:
                st.error("Please upload a PDF file.")
        else:
            st.info("Please upload a CSV file.")

if __name__ == "__main__":
    main()
