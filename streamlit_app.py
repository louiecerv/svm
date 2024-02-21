import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():
    # Display the DataFrame with formatting
    st.title("Support Vector Machine Classifier")
    text = """Louie F. Cervantes, M.Eng. \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.subheader('Description')

    text = """
        Replace with description of SVM. """
    st.write(text)
    if st.button('Start'):
        X, y = make_blobs(n_samples=200, centers=2, center_box=(-10, 10), cluster_std=1.0, random_state=42)     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

        clfSVM = svm.SVC(kernel='linear', C=1000)
        clfSVM.fit(X_train, y_train)
        y_test_pred = clfSVM.predict(X_test)

        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))

        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.write(cm)
        
        st.subheader('Visualization')

        #predict the class of new data
        newdata = [[3,4], [5,6]]

        # Create the figure and axes object
        fig, ax = plt.subplots(figsize=(9, 9))

        # Scatter plot of the data
        ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

        # Predict the class of new data
        st.text(f'predicted classes: {clfSVM.predict(newdata)}')

        # Plot the decision function directly on ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clfSVM.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '--', '--'])

        # Plot support vectors
        ax.scatter(clfSVM.support_vectors_[:, 0], clfSVM.support_vectors_[:, 1], s=100, linewidth=1, facecolor='none')

        st.pyplot(fig)


if __name__ == "__main__":
    app()
