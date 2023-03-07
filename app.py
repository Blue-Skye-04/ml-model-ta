from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
with open('finalized_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)
    
app = Flask(__name__)

@app.route('/', methods = ["GET","POST"])
def marks():
    if request.method == "POST":
        uraian = request.form["uraian"]
        simple_test = [uraian]
        simple_test_dtm = vectorizer.transform(simple_test)
        prediction = model.predict(simple_test_dtm)
        pred = prediction
    
    return render_template("index.html", prediction = pred)    
    

if __name__ == "__main__":
    app.run(debug=True)