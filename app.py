from flask import Flask, render_template, request
import pickle
import os

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = vectorizer.transform([news])
        prediction = model.predict(data)[0]
        return render_template('index.html', prediction=prediction, news=news)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Render assigns a PORT env variable
    app.run(host='0.0.0.0', port=port)
