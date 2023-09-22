import os
import requests
import string

from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz

import nltk
from nltk import WordNetLemmatizer, PorterStemmer
nltk.download('wordnet')

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Configs:
    # 0.3+
    def __init__(self) -> None:
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = 'cpu'
        print('DEVICE: ', self.device)

        self.BASE_MODEL = 'facebook/bart-base'
        self.MAX_TOKENS = 1024

        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.BASE_MODEL, num_labels=1)
        self.model.load_state_dict(torch.load(os.path.join('model', 'BART_best.pth')))
        self.model.to(self.device)

        self.lemmatizer = WordNetLemmatizer()
        self.porter = PorterStemmer()

    def preprocess_text(self, data: str) -> str:
            text: str = data.strip()
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = self.lemmatizer.lemmatize(text, 'v')
            text = self.porter.stem(text)
            return text
        

configs = Configs()

app = Flask(__name__)
# CORS(app)


def getPDF(response):
    name = 'sample.pdf'

    pdf_url = response['url']
    pdf_resp = requests.get(pdf_url)

    with open(name, 'wb') as f: 
        f.write(pdf_resp.content)

    text= ''
    with fitz.open(name) as doc:
        for page in doc: text += page.get_text()

    return text


def predict(text: str, config: Configs):
    text = configs.preprocess_text(text)
    inputs = configs.tokenizer(
        text, 
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=config.MAX_TOKENS,
        return_tensors="pt"
    )
    print(inputs)
    inputs.to(configs.device)
    with torch.inference_mode():
        outputs = configs.model(**inputs)
    
    logits = outputs['logits']
    probablity = torch.sigmoid(logits) + 0.3
    print(f'------------------------------------- OUTPUTS: ------------------------------- {probablity}')
    return float(probablity.squeeze())


@app.route("/petitionSuccessProb", methods=['POST'])
def is_alive():
    response = request.get_json()
    petitionId = response['petitionId']
    text = getPDF(response)
    pred_prob = predict(text, configs)


    print(response)
    return jsonify({
        'petitionId': petitionId,
        'prediction_prob': f'{pred_prob:.2f}',
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)


