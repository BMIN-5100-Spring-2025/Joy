import csv
import os
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_database(database_embedding):
    with open(database_embedding, "r", encoding="utf-8") as f:
        database_eb = json.load(f)
    return database_eb

def write_csv(output_file, top_3):
    with open(output_file, mode='w', encoding='utf-8') as outfile:
        json.dump(top_3, outfile, ensure_ascii=False, indent=4)

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
    database_embedding = "../data/input/disease_embedding.json"
    database_eb = get_database(database_embedding)

    base_directory = os.path.dirname(os.path.dirname(__file__))

    # input_directory = os.getenv('INPUT_DIR', os.path.join(base_directory, 'data/input/'))
    output_directory = os.getenv('OUTPUT_DIR', os.path.join(base_directory, 'data/output/'))
    output_file = os.path.join(output_directory, 'result.json')

    top_3 = calculate_sim(model, database_eb, user_input)
    write_csv(output_file, top_3)