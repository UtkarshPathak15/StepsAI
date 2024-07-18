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
import socket
import time

def extract_main_content(soup):

  main_content = soup.find('div', itemprop='articleBody')
  if not main_content:
    main_content = soup.find('article', id='contents')
    if not main_content:
      return ""

  for tag in main_content.find_all(['h1', 'h2', 'h3']):
    tag.decompose()

  return main_content.get_text(separator='\n', strip=True)

def crawl(start_url, max_depth=5):
    visited = set()
    queue = deque([(start_url, 1)])
    all_data = []

    while queue:
        url, depth = queue.popleft()
        if depth > max_depth or url in visited:
            continue
        visited.add(url)

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        page_text = extract_main_content(soup)

        page_data = {
            'url': url,
            'depth': depth,
            'text': page_text
        }

        all_data.append(page_data)

        links = soup.find_all('a', href=True)
        for link in links:
            sub_url = urljoin(url, link['href'])
            main_url, _ = urldefrag(sub_url)
            if main_url.startswith('https://docs.nvidia.com/cuda/') and main_url not in visited:
                queue.append((main_url, depth + 1))

    return all_data


def chunk_data(data):
    scraped_data = []
    c=0
    for i in range(len(data)):
        text1 = data[i]['text']
        # text1 = data[i]
        text1 = text1.split('\uf0c1')
        text1 = [item.split('.\n') for item in text1]
        text1 = [item for sublist in text1 for item in sublist]
        text1 = [item.replace('\n'," ") for item in text1]
        text1 = [re.sub(r'\s+', ' ', item) for item in text1]
        for text in text1:
            if text!='' and len(text.split())>10:
                scraped_data.append(text)

    scraped_data = [s for s in scraped_data if len(s) <= 2000 and len(s)>100]
    return scraped_data


def preprocess_text(data,stop_words,punctuation,stemmer):
    p_data = []
    for text in data:
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.lower() not in punctuation]
        tokens = [token for token in tokens if token.lower() not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        p_data.append(" ".join(tokens))

    return p_data

def is_server_running(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


def connect_to_milvus():
    # connections.disconnect("default")
    if not is_server_running(host="127.0.0.1",port=default_server.listen_port):
        default_server.start()
    connections.connect("default",host="127.0.0.1",port=default_server.listen_port)
    # connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus.")
    

def create_collection(collection_name, dim=384):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(fields, "CUDA documentation collection")
    collection = Collection(collection_name, schema)
    print(f"Collection schema: {collection.schema}")
    return collection


def create_index(collection, field_name="embedding"):
    index= {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index(field_name, index)
    print(f"Index created on field '{field_name}' successfully!")
    

def insert_data(collection,scraped_data,embeddings):
    batch_size = 1000
    for i in range(0, len(scraped_data), batch_size):
        end = min(i + batch_size, len(scraped_data))
        batch_entities = [
            list(range(i,end)),
            [text for text in scraped_data[i:end]],
            [embedding for embedding in embeddings[i:end]]
            ]
        collection.insert(batch_entities)
    collection.flush()
    print("Data inserted successfully!")
    

def normalize_distances(entities):
    distances = [entity[1].distance for entity in entities]
    scores = [entity[0] for entity in entities]
    min_dist = min(distances)
    max_dist = max(distances)

    min_score = min(scores)
    max_score = max(scores)

    normalized_distances = [(dist - min_dist) / (max_dist - min_dist) for dist in distances]
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]

    for i, entity in enumerate(entities):
        entity[1].distance = normalized_distances[i]
        entity[0] = normalized_scores[i]

    return entities


def hybrid_retrieval(query_inp,collection,model,preprocessed_sentences,scraped_data,alpha=0.5):
    tokenized_corpus = [text.split() for text in preprocessed_sentences]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query_inp[0].split()
    doc_scores = bm25.get_scores(tokenized_query)

    doc_scores1 = []
    dict_scores = {}
    for i,v in enumerate(doc_scores):
        doc_scores1.append((i,v))
        dict_scores[i]=v

    doc_scores1_sorted = sorted(doc_scores1, key=lambda x: x[1],reverse=True)[:300]

    # docs = bm25.get_top_n(tokenized_query, list(scraped_data), n=100)
    # print(docs[:5],sep="\n\n")
    query_embedding = model.encode(query_inp, convert_to_tensor=True,show_progress_bar=True)
    query_embedding = query_embedding.cpu().data.numpy()
    collection.load()
    # print(collection.num_entities)
    bert_result = collection.search(query_embedding,"embedding",param={"metric_type":"COSINE", "params":{"nprobe":128}},limit=1500,output_fields=["index","text","embedding"])
    # print(bert_result[0][:5])
    bert_results_final = [r.entity.get('index') for r in bert_result[0]]
    scores_final = []
    for i in doc_scores1_sorted:
        if(i[0] in bert_results_final):
            scores_final.append(i[0])

    bert_results1 = []
    for r in bert_result[0]:
        if r.entity.get('index') in scores_final:
            bert_results1.append([dict_scores[r.entity.get('index')],r])

    # print(bert_results1)

    bert_results1 = normalize_distances(bert_results1)
    for i in range(len(bert_results1)):
        bert_results1[i][0] = alpha*bert_results1[i][0]+(1-alpha)*bert_results1[i][1].distance

    bert_results_final = sorted(bert_results1,key=lambda x: x[0],reverse=True)
    bert_results_final = [r[1].entity.get('text') for r in bert_results_final]
    
    return bert_results_final


def question_answer(user_query, context, model, tokenizer):
    
    inputs = tokenizer.encode_plus(user_query, context, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']

    outputs = model(input_ids, token_type_ids=token_type_ids)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits)

    answer_tokens = input_ids[0][answer_start:answer_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    corrected_answer = answer.capitalize()

    return corrected_answer