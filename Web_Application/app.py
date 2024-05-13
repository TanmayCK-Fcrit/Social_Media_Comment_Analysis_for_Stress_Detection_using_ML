# Importing necessary libraries and functions
from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import nltk
import re
from nltk.corpus import stopwords
import string
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Download necessary resources from NLTK
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# Define data cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Load the dataset and preprocess the data
data = pd.read_csv(r"new_stress.csv")
data["text"] = data["text"].apply(clean)
data["label"] = data["label"].map({0: "  Stress/  Depression", 1: "Stress/Depression"})

x = np.array(data["text"])
y = np.array(data["label"])

# Vectorize the text data
cv = CountVectorizer()
X = cv.fit_transform(x)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

# Train Bernoulli Naive Bayes model
model_bnb = BernoulliNB()
model_bnb.fit(xtrain, ytrain)

# Train Support Vector Machine (SVM) model
model_svm = SVC(kernel='linear')
model_svm.fit(xtrain, ytrain)

# Train Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(xtrain, ytrain)

# Function to predict text using a specific model
def predict_text(model, user_input):
    data = cv.transform([user_input])
    output = model.predict(data)
    return output[0]

# Function to generate a heatmap of classification report
def generate_classification_report_heatmap(y_pred, model_name):
    report_dict = classification_report(ytest, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df, annot=True, cmap='coolwarm', cbar=True)
    
    plt.title(f"{model_name} Classification Report Heatmap")
    plt.xlabel("Metrics")
    plt.ylabel("Classes")

    # Save plot to a bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return img_data

# Function to generate a confusion matrix as a heatmap
def generate_confusion_matrix_heatmap(y_pred, model_name):
    cm = confusion_matrix(ytest, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save plot to a bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return img_data

# Define the route for the root URL
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("text_entry")
        model_choice = request.form.get("model_choice")
        
        if model_choice == "BernoulliNB":
            model = model_bnb
        elif model_choice == "SVM":
            model = model_svm
        elif model_choice == "Random Forest":
            model = model_rf
        else:
            flash("Invalid model choice!", "danger")
            return redirect(url_for("index"))
        
        prediction = predict_text(model, user_input)
        flash(f"The model predicts: {prediction}", "success")
        
    return render_template("index.html")

# Define the route to display classification report heatmap
@app.route("/classification_report_heatmap/<model_name>")
def classification_report_heatmap(model_name):
    if model_name == "BernoulliNB":
        model = model_bnb
    elif model_name == "SVM":
        model = model_svm
    elif model_name == "Random Forest":
        model = model_rf
    else:
        flash("Invalid model choice!", "danger")
        return redirect(url_for("index"))
    
    y_pred = model.predict(xtest)
    img_data = generate_classification_report_heatmap(y_pred, model_name)
    return render_template("report_heatmap.html", model_name=model_name, img_data=img_data)

# Define the route to display confusion matrix as a heatmap
@app.route("/confusion_matrix/<model_name>")
def confusion_matrix_heatmap(model_name):
    if model_name == "BernoulliNB":
        model = model_bnb
    elif model_name == "SVM":
        model = model_svm
    elif model_name == "Random Forest":
        model = model_rf
    else:
        flash("Invalid model choice!", "danger")
        return redirect(url_for("index"))
    
    y_pred = model.predict(xtest)
    img_data = generate_confusion_matrix_heatmap(y_pred, model_name)
    return render_template("confusion_matrix.html", model_name=model_name, img_data=img_data)

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
