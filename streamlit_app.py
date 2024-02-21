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
        Replace this with description of SVM. """
    st.write(text)

    # Get user's inputs
    n_samples = int(st.number_input("Enter the number of samples:"))
    cluster_std = st.number_input("Standard deviation (between 0 and 1):")
    random_state = int(st.number_input("Random seed (between 0 and 100):"))
    n_clusters = int(st.number_input("Number of Clusters (between 2 and 6):"))
    
    if st.button('Start'):
        centers = generate_random_points_in_square(-4, 4, -4, 4, n_clusters)
        X, y = make_blobs(n_samples=n_samples, n_features=2,
                        cluster_std=cluster_std, centers = centers,
                        random_state=random_state)
                   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clfSVM = svm.SVC(kernel='linear', C=1000)
        clfSVM.fit(X_train, y_train)
        y_test_pred = clfSVM.predict(X_test)

        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))

        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.write(cm)
        if n_clusters == 2:
            st.subheader('Visualization')
    
            #use the Numpy array to merge the data and test columns
            dataset = np.column_stack((X, y))

            df = pd.DataFrame(dataset)
            # Add column names to the DataFrame
            df = df.rename(columns={0: 'X', 1: 'Y', 2: 'Class'})
            # Extract data and classes
            x = df['X']
            y = df['Y']
            classes = df['Class'].unique()

            # Create the figure and axes object
            fig, ax = plt.subplots(figsize=(9, 9))
    
            # Scatter plot of the data
            ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = clfSVM.decision_function(xy).reshape(XX.shape)
    
            ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '--', '--'])
    
            # Plot support vectors
            ax.scatter(clfSVM.support_vectors_[:, 0], clfSVM.support_vectors_[:, 1], s=100, linewidth=1, facecolor='none')
    
            st.pyplot(fig)

def generate_random_points_in_square(x_min, x_max, y_min, y_max, num_points):
    """
    Generates a NumPy array of random points within a specified square region.

    Args:
        x_min (float): Minimum x-coordinate of the square.
        x_max (float): Maximum x-coordinate of the square.
        y_min (float): Minimum y-coordinate of the square.
        y_max (float): Maximum y-coordinate of the square.
        num_points (int): Number of points to generate.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (num_points, 2) containing the generated points.
    """

    # Generate random points within the defined square region
    points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(num_points, 2))

    return points

if __name__ == "__main__":
    app()
