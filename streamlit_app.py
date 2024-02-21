import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    # Display the DataFrame with formatting
    st.title("Spam Detection using the Naive Bayes Classifier")
    text = """Louie F. Cervantes, M.Eng. \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.subheader('Description')

    text = """The SMS Spam Collection Dataset from Kaggle, often used 
        to demonstrate Naive Bayes as a spam detector, is a valuable 
        resource for machine learning enthusiasts and researchers 
        interested in text classification, particularly spam 
        filtering. Here's a breakdown of its key characteristics:"""
    st.write(text)
    st.write('Data:')
    st.write('Size: 5,572 SMS messages in English')
    st.write("""Format: Plain text, with each line containing two columns: 
            label ("ham" for legitimate, "spam" for spam) and message content.
            Content: The messages originate from various sources, 
            including Singaporean students and general English speakers. 
            They cover a diverse range of topics and communication styles.""")

    data = pd.read_csv('spam.csv', 
                        dtype='str', header=0, 
                        sep = ",", encoding='latin')        

    # display the dataset
    st.header('The Dataset')
    
    st.dataframe(data, use_container_width=True)  
    X = data['v2']
    y = data['v1']        
    
    # Create a new figure and axes object
    fig, ax = plt.subplots()

    # Create a horizontal barplot using seaborn
    sns.countplot(y='v1', data=data, hue='v1', palette='bright', ax=ax)

    # Set title
    ax.set_title('Plot of Spam/Ham Distribution')
    st.pyplot(fig)

    clfNB = make_pipeline(TfidfVectorizer(), MultinomialNB())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    clfNB.fit(X_train, y_train)
    
    y_test_pred = clfNB.predict(X_test)
    if st.button('Start'):
        
        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))

        st.subheader('Confusion Matrix')
        cmNB = confusion_matrix(y_test, y_test_pred)
        st.write(cmNB)

        st.subheader('Sample predictions')
        text = 'receive a free entry'
        st.text(text + ' ---> ' + predict_category(clfNB, text))
        text = 'you could win a prize'
        st.text(text + ' ---> ' + predict_category(clfNB, text))
        text = 'We will have a meeting'
        st.text(text + ' ---> ' + predict_category(clfNB, text))
        text = 'camera for free'
        st.text(text + ' ---> ' + predict_category(clfNB, text))

    strinput = st.text_input("Enter message:")
    if st.button('Submit'):
        st.write('The message is : ' + predict_category(clfNB, strinput))

def predict_category(clf, s):
    pred = clf.predict([s])
    return str(pred[0])

if __name__ == "__main__":
    app()
