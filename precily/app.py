import json
from flask import Flask, request, jsonify
import similarity_score

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process():
    data = request.get_json()
    text1 = data['text1']
    text2 = data['text2']
    
    
    
    score = similarity_score.calculate_similarity_score(text1,text2)
    print(score)
    result = {"similarity score":score.item()}
    return jsonify(result)
if __name__ =='__main__':
    app.run()