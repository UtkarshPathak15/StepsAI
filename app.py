import streamlit as st
from required_functions import *
import os
import re
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sentence_transformers import SentenceTransformer, util
import torch
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from milvus import default_server
from rank_bm25 import BM25Okapi
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag
from collections import deque
import json
from pymilvus import utility
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

@st.cache_resource
def start():
    
    torch.manual_seed(15)
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()

    if not os.path.exists('crawled_data.json'):
        url = 'https://docs.nvidia.com/cuda/'
        data = crawl(url, max_depth=5)
        
        with open('crawled_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        with open('crawled_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
    scraped_data = chunk_data(data)
    scraped_data = list(set(scraped_data))
    preprocessed_sentences = preprocess_text(scraped_data,stop_words,punctuation,stemmer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    model2 = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer2 = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model.to(device)
    model2.to(device)

    if not os.path.exists('embeddings1.npy'):
        embeddings1 = model.encode(preprocessed_sentences, convert_to_tensor=True,show_progress_bar=True)
        # torch.save(model,"model.pt")
        embeddings1 = embeddings1.cpu().data.numpy()
        np.save('embeddings1.npy',embeddings1)
    else:
        embeddings1 = np.load('embeddings1.npy')

    connect_to_milvus()
    collection = ""
   
    if not utility.has_collection("cuda_docs_collection"):
        collection = create_collection("cuda_docs_collection") 
    else:
        collection = Collection("cuda_docs_collection")
        collection.drop()
        collection = create_collection("cuda_docs_collection")
        

    create_index(collection)
    insert_data(collection,scraped_data,embeddings1)
    # if not collection.has_index():
    #     create_index(collection)
    
    # num_entities = collection.num_entities
    # if num_entities==0:
    #     insert_data(collection,scraped_data,embeddings1)  

    return stop_words,punctuation,stemmer,collection,model,preprocessed_sentences,scraped_data,model2,tokenizer2

def main():
    stop_words,punctuation,stemmer,collection,model,preprocessed_sentences,scraped_data,model2,tokenizer2 = start()
    st.title("Model Query Interface")
    user_query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if user_query:
            print(user_query)

            query_inp = preprocess_text([user_query],stop_words,punctuation,stemmer)
            # print(query_inp)
            retrieved_docs = hybrid_retrieval(query_inp,collection,model,preprocessed_sentences,scraped_data,alpha=0.6)
            output = retrieved_docs[:5]
            context = "\n".join(output)
            # print(output)
            answer = question_answer(user_query,context,model2,tokenizer2)
            st.write("Answer: ")
            st.markdown(f"{answer}")
            st.markdown("""---""")
            st.write("Retrieved Text (For reference):")
            for i, otp in enumerate(output, start=1):
                st.markdown(f"**{i}.** {otp}")
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()