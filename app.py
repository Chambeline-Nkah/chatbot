from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Loading the label mapping
with open('./label_mapping.json', 'r') as file:
    label_mapping = json.load(file)

# Loading the BERT model
model = BertForSequenceClassification.from_pretrained('./model')
tokenizer = AutoTokenizer.from_pretrained('./model')

# Load the semantic model
semantic_model = SentenceTransformer('./sent_transf')
response_embeddings = np.load('./response_embeddings.npy')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def semantic_search(query, top_k=5):
    query_embedding = semantic_model.encode([query])
    similarities = cosine_similarity(query_embedding, response_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(list(label_mapping.keys())[i], similarities[i]) for i in top_indices]

def predict_category(input_text):
    cleaned_input = clean_text(input_text)
    inputs = tokenizer(f"{cleaned_input}", return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    bert_prediction = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class_id)]
    semantic_results = semantic_search(f"{cleaned_input}")
    if bert_prediction in [result[0] for result in semantic_results]:
        return bert_prediction
    else:
        return semantic_results[0][0]

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('input')
    if not input_text:
        return jsonify({'error': 'Invalid input'}), 400
    response = predict_category(input_text)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
