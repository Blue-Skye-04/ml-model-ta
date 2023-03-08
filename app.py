from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
with open('finalized_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)
    
app = Flask(__name__)
def takeSecond(elem):
    return elem[1]

@app.route('/', methods = ["GET","POST"])
def index():
    pred = ""
    if request.method == "POST":
        uraian = request.form["uraian"]
        simple_test = [uraian]
        simple_test_dtm = vectorizer.transform(simple_test)
        prediction = model.predict_proba(simple_test_dtm)

        pred = [["Sistem Energi Listrik (SEL)", prediction[0][0]],["Sistem Mekatronika (SM)", prediction[0][1]],["Teknologi Informasi dan Komunikasi (ICT)", prediction[0][2]]]
        pred.sort(key=takeSecond)
    return render_template("index.html", prediction = pred, text = uraian)    
    

if __name__ == "__main__":
    app.run(debug=True)