import os
import numpy as np
import pandas as pd
import pickle
import base64
import streamlit as st
import string
from nltk import pos_tag
import plotly.graph_objects as go
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

stopwords = nltk.corpus.stopwords.words('english')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# ---------------Functions to clean the input text--------------#

def tokenize_remove_punctuations(text):
    clean_text = []
    text = text.split(" ")
    for word in text:
        word = list(word)
        new_word = []
        for c in word:
            if c not in string.punctuation:
                new_word.append(c)
        word = "".join(new_word)
        if len(word) > 0:
            clean_text.append(word)
    return clean_text


def remove_stopwords(text):
    clean_text = []
    for word in text:
        if word not in stopwords:
            clean_text.append(word)
    return clean_text


def pos_tagging(text):
    tagged = nltk.pos_tag(text)
    return tagged


def get_wordnet(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(pos_tags):
    lemmatized_text = []
    for t in pos_tags:
        word = WordNetLemmatizer().lemmatize(t[0], get_wordnet(t[1]))
        lemmatized_text.append(word)
    return lemmatized_text


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = tokenize_remove_punctuations(text)
    text = [word for word in text if not any(c.isdigit() for c in word)]
    text = remove_stopwords(text)
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tagging(text)
    text = lemmatize(pos_tags)
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return text


def transform_input(text):
    text = np.array([text])
    text = pd.Series(text)
    text = clean_text(text)
    return text


# ---------------Loading trained models--------------#

current_dir = os.path.dirname(os.path.abspath(__file__))
vectorizer_path = os.path.join(current_dir, "Pickle Files", "vectorizer.pkl")
current_dir = os.path.dirname(os.path.abspath(__file__))
linear_svc_path = os.path.join(current_dir, "Pickle Files", "LinearSVC.pkl")
multinomial_nb_path = os.path.join(current_dir, "Pickle Files", "MultinomialNB.pkl")
logistic_regression_path = os.path.join(current_dir, "Pickle Files", "LogisticRegression.pkl")
kneighbors_classifier_path = os.path.join(current_dir, "Pickle Files", "KNeighborsClassifier.pkl")

pickle_in = open(vectorizer_path, "rb")
vect = pickle.load(pickle_in)

pickle_in = open(linear_svc_path, "rb")
LinearSVC = pickle.load(pickle_in)

pickle_in = open(multinomial_nb_path, "rb")
MultinomialNB = pickle.load(pickle_in)

pickle_in = open(logistic_regression_path, "rb")
LogisticRegression = pickle.load(pickle_in)

pickle_in = open(kneighbors_classifier_path, "rb")
KNeighborsClassifier = pickle.load(pickle_in)


def login():
    st.markdown("<h1 style='font-size: 35px;'></h1>", unsafe_allow_html=True)
    st.title("Cyberbullying Detection using Ml")
    username = st.text_input("Username", key="username", value="")
    password = st.text_input("Password", key="password", value="", type="password")

    if st.button("Login"):
        # Your login logic here
        return True
    else:
        return False


def display_content():
    # ---------------Creating selectbox--------------#
    st.markdown("<h1 style='font-size: 35px;'></h1>", unsafe_allow_html=True)
    st.title("Cyberbullying Detection using Ml")
    option = st.selectbox(
        'Select Model',
        ('Select', 'Linear SVC', 'K-Nearest Neighbour', 'Logistic Regression', 'Multinomial Naive Bayes'),
        key='selectbox',
        help="Choose the model"
    )

    # ---------------Input--------------#

    comment = st.text_input("Enter any comment", " ", key='text-input')
    comment = clean_text(comment)
    comment = np.array([comment])
    comment = pd.Series(comment)
    comment = vect.transform(comment)
    pred = [0]  # creating a list to store the output value

    # ---------------Predicting output--------------#

    if st.button("Predict"):
        st.markdown("""
            <style>
            .prediction-text {
                font-size: 24px;
            }
            </style>
        """, unsafe_allow_html=True)

        if pred[0] == 1:
            st.markdown('<p class="prediction-text">prediction: Bullying comment!!!!</p>', unsafe_allow_html=True)
        #else:
        #    st.markdown('<p class="prediction-text">prediction: Normal comment.</p>', unsafe_allow_html=True)
        if option == 'Multinomial Naive Bayes':
            pred = MultinomialNB.predict(comment)
            if pred[0] == 1:
                st.success('prediction: {}'.format("Bullying comment!!!!"))
            else:
                st.success('prediction: {}'.format("Normal comment."))

        elif option == 'Linear SVC':
            pred = LinearSVC.predict(comment)
            if pred[0] == 1:
                st.success('prediction: {}'.format("Bullying comment!!!!"))
            else:
                st.success('prediction: {}'.format("Normal comment."))

        elif option == 'K-Nearest Neighbour':
            pred = KNeighborsClassifier.predict(comment)
            if pred[0] == 1:
                st.success('prediction: {}'.format("Bullying comment!!!!"))
            else:
                st.success('prediction: {}'.format("Normal comment."))

        elif option == 'Logistic Regression':
            if pred[0] == 1:
                st.success('prediction: {}'.format("Bullying comment!!!!"))
            else:
                st.success('prediction: {}'.format("Normal comment."))
        else:
            st.write("You haven't selected any model :(")


def display_performance_analysis():
    # Load the performance data and accuracies
    performance_data = os.path.join(current_dir, "Pickle Files", "results.pkl")
    pickle_in = open(performance_data, "rb")
    results = pickle.load(pickle_in)

    # Display the performance analysis
    st.markdown("<h1>Performance Analysis</h1>", unsafe_allow_html=True)

    # Display the performance data in a table
    st.markdown("<h2>Performance Data</h2>", unsafe_allow_html=True)
    st.dataframe(results)

    # Display the accuracies using a bar chart
    st.markdown("<h2>Accuracies</h2>", unsafe_allow_html=True)
    accuracies = results.set_index('Algorithm')['Accuracy Score : Test']
    fig = go.Figure(data=[go.Bar(x=list(accuracies.index), y=list(accuracies.values))])
    fig.update_layout(title="Model Accuracies", xaxis_title="Model", yaxis_title="Accuracy")
    st.plotly_chart(fig)


def main():
    page = st.sidebar.selectbox("Login or Display", ("Login", "Display", "Performance Analysis"))

    if page == "Login":
        if login():
            st.sidebar.success("Login successful!")
            st.sidebar.markdown("### Navigation")
            st.sidebar.markdown("Select 'Display' to view the content.")
        else:
            st.sidebar.error("Login failed. Please try again.")
    elif page == "Display":
        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://assets.nst.com.my/images/articles/hacker-3342696_1920_1614744249.jpg");
        background-size: 180%;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: local;
        }}
        [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        }}
        [data-testid="stToolbar"] {{
        right: 2rem;
        }}
        </style>
        """

        st.markdown(page_bg_img, unsafe_allow_html=True)
        display_content()
    elif page == "Performance Analysis":
        display_performance_analysis()


if __name__ == "__main__":
    main()
