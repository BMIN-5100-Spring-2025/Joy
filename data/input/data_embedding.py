import csv
from sentence_transformers import SentenceTransformer
import pandas as pd
import json

database_file = r"D:\Users\Joy\Desktop\Joy\data\input\disease_symptoms.json"
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
with open(database_file, 'r', encoding="utf-8") as f:
    disease_db = json.load(f)

disease_embedding = {}

for disease, symptoms in disease_db.items():
    embeddings = model.encode(["".join(symptoms)])
    disease_embedding[disease] = embeddings.tolist()

with open("disease_embedding.json", 'w', encoding='utf-8') as f:
    json.dump(disease_embedding, f, ensure_ascii=False, indent=4)