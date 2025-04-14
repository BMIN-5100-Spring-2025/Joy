import csv
import os
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import boto3

def get_database(database_embedding, mode):
    if mode == 'local':
        with open(database_embedding, "r", encoding="utf-8") as f:
            database_eb = json.load(f)
        return database_eb
    elif mode == 's3':
        os.makedirs(os.path.dirname(database_embedding), exist_ok=True)
        s3.download_file('diseasepredictor2025', 'data/input/disease_embedding.json', database_embedding)
        with open(database_embedding, "r", encoding="utf-8") as f:
            database_eb = json.load(f)
        return database_eb

def write_result(output_file, top_3, mode):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode='w', encoding='utf-8') as outfile:
            json.dump(top_3, outfile, ensure_ascii=False, indent=4)
    if mode == 's3':
        object_name = os.path.join('data/output', 'result.json')
        s3.upload_file(output_file, 'diseasepredictor2025', object_name)

def calculate_sim(model, database_eb, user_input):
    user_eb = model.encode(user_input)
    similarity = {}

    for disease, eb in database_eb.items():
        similarity[disease] = cosine_similarity(user_eb.reshape(1,-1), np.array(eb).reshape(1,-1))
    top_3 = sorted(similarity.items(), key=lambda item: item[1], reverse=True)[:3]
    top_3_final = [(key, value[0][0]) for key, value in top_3]
    return top_3_final

if __name__ == "__main__":
    user_input = ["cough, fever, and sour throat"]
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    # database_embedding = "../data/input/disease_embedding.json"
    s3 = boto3.client("s3")
    mode = os.getenv('MODE', 's3')
    input_directory = os.getenv('INPUT_DIR', '/data/input')
    output_directory = os.getenv('OUTPUT_DIR', '/data/output')
    output_file = os.path.join(output_directory, 'result.json')
    database_embedding = os.path.join(input_directory, 'disease_embedding.json')
    database_eb = get_database(database_embedding, mode)

    top_3 = calculate_sim(model, database_eb, user_input)
    write_result(output_file, top_3, mode)