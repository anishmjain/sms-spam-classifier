import pickle
import string
from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the saved models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        
        if not input_sms.strip():
            return render_template('index.html', error="Please enter a valid message.")
        
        try:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            
            # 3. predict
            result = model.predict(vector_input)[0]
            
            # 4. Display result
            if result == 1:
                prediction = "Spam"
            else:
                prediction = "Not Spam"

            return render_template('result.html', prediction=prediction)
        
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
