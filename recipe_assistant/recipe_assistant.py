import re
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import torch
from dotenv import load_dotenv
import os
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

COLD_START = False
LOAD_FAISS = True
n_k_relevants = 1

# 1. LOAD AND VECTORIZE OUR RECIPE DATA SET DOCUMENTS:
loader = CSVLoader(file_path="./recipe_assistant/recipes.csv", encoding='utf-8')
documents = loader.load()

# WITH OPEN SOURCE sentence_transformers embedding using LANGCHAIN WRAPPER ABD FAISS
# model_name = "hkunlp/instructor-xl"
model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

model_embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, 
                                            model_kwargs=model_kwargs, 
                                            encode_kwargs=encode_kwargs)

if COLD_START:
    print("\n")
    print("We have to create the whole data base of embeddings")
    print("\n")
    print("ACTING FAISS....")
    db_instructEmbedd = FAISS.from_documents(documents, model_embeddings)
    db_instructEmbedd.save_local("./recipe_assistant/faiss_index")
    retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": n_k_relevants})
elif LOAD_FAISS:
    print("\n")
    print("LOADING... DB EMBEDDINGS")
    print("\n")
    db_instructEmbedd = FAISS.load_local("./recipe_assistant/faiss_index", model_embeddings)
    retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": n_k_relevants})    
else:
    pass


# DEFINE PROMPT TEMPLATE
template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You have to name the list of ingredients at the begining of the response and substract in bullet points the ordered instructions to do the recipe.

### Input:
{recipe}

### Response:

"""


prompt = PromptTemplate(
    input_variables=["recipe"],
    template=template
)


config = {'temperature': 0.1, 'max_new_tokens': 400}
openLLM = CTransformers(model = "TheBloke/vicuna-7B-v1.5-GGUF", model_file = 'vicuna-7b-v1.5.Q5_K_M.gguf', config=config)


# Define Chain that is the LLM and the propmt. that comes with two inputs:
chain = LLMChain(llm=openLLM, prompt=prompt)

#Retrieval function:
def retrieve_info(query):
    similar_response = retriever.get_relevant_documents(query)
    page_contents_array = [re.sub("Recipe: ", "", doc.page_content) for doc in similar_response]
    return page_contents_array

#Generate response:
def generate_response(message):
    best_recipe = retrieve_info(message)
    response = chain.run(recipe=best_recipe)
    return response

# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Old book recipes", page_icon=":pizza:")

    st.header("Recipe generator :pizza:")
    message = st.text_area("write whished dish here")

    if message:
        st.write("Generating best recipe dish based on your wish...")

        result = generate_response(message)

        st.info(result)
    LOAD_FAISS = False

if __name__ == '__main__':
    main()