import csv
import os
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import boto3
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # 处理跨域请求 (如果前端和后端不在同一个域名)

# app = Flask(__name__)
# CORS(app) # 允许所有来源的跨域请求 (生产环境需要配置更严格的策略)

BUCKET      = 'diseasepredictor2025'
SESSION_ID  = os.getenv('SESSION_ID') 
s3 = boto3.client("s3") # 确保 s3 client 在函数内部创建或在全局作用域已存在

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
    
def get_user_symptom(mode):
    if mode == 'local':
        local = './data/input/symptom.json' 
    else:
        # !!!!!
        key = f'data/input/test-123/symptom.json'
        local  = '/tmp/symptom.json'
        s3.download_file(BUCKET, key, local)
    with open(local, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return data.get('symptom', '')

def write_result(output_file, top_3, mode):
    if mode == 'local':
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        local = output_file
    else:
        local = '/tmp/result.json'
    with open(local, 'w', encoding='utf-8') as f:
        json.dump(top_3, f, ensure_ascii=False, indent=4)
    if mode == 's3':
        # with open(local, 'w', encoding='utf-8') as f:
        #     json.dump(top_3, f, ensure_ascii=False, indent=4)
        #!!!!!!!!!!!!!!!!
        object_name = f'data/output/test-123/result.json'
        s3.upload_file(local, 'diseasepredictor2025', object_name)

def calculate_sim(model, database_eb, user_input_list): # 接收用户输入列表
    if not user_input_list:
        return []
    user_input = user_input_list[0] # 假设前端只发送一个症状描述
    user_eb = model.encode(user_input)
    similarity = {}

    for disease, eb in database_eb.items():
        similarity[disease] = cosine_similarity(user_eb.reshape(1,-1), np.array(eb).reshape(1,-1))
    top_3 = sorted(similarity.items(), key=lambda item: item[1], reverse=True)[:3]
    top_3_final = [(key, value[0][0]) for key, value in top_3]
    return top_3_final

def handler(_event=None, _ctx=None):
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    mode = os.getenv('MODE', 's3')
    input_directory = os.getenv('INPUT_DIR', '/data/input')
    output_directory = os.getenv('OUTPUT_DIR', '/data/output')
    output_file = os.path.join(output_directory, 'result.json')
    database_embedding = os.path.join(input_directory, 'disease_embedding.json')
    # Fargate 容器入口（不用再跑 Flask）
    database_eb = get_database(database_embedding, mode)
    symptom  = get_user_symptom(mode)

    if not symptom:
        print('symptom empty'); return
    top_3  = calculate_sim(model, database_eb, [symptom])
    write_result(output_file, top_3, mode)

if __name__ == '__main__':
    handler()

# @app.route('/api/predict', methods=['POST'])
# def predict_disease():
#     data = request.get_json()
#     user_input_list = data.get('user_input')

#     model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
#     mode = os.getenv('MODE', 's3')
#     input_directory = os.getenv('INPUT_DIR', '/data/input')
#     output_directory = os.getenv('OUTPUT_DIR', '/data/output')
#     output_file = os.path.join(output_directory, 'result.json')
#     database_embedding = os.path.join(input_directory, 'disease_embedding.json')
#     database_eb = get_database(database_embedding, mode)

#     top_3 = calculate_sim(model, database_eb, user_input_list)
#     write_result(output_file, top_3, mode)

#     return jsonify({'predictions': top_3}) # 返回预测结果给前端

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000) # 启动 Flask 应用