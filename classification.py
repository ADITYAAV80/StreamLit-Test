import streamlit as st

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier


@st.cache_data
def load_data():

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    df['species'] = iris.target

    return df,iris.target_names

df,target_name = load_data()

model = RidgeClassifier()

model.fit(df.iloc[:,:-1],df['species'])

st.subheader("Sample data")
st.write(df)

st.sidebar.title("Input features")

sl = st.sidebar.slider("Sepal Lengh", 0,10)
sw = st.sidebar.slider("Sepal Width", 0,10)
pl = st.sidebar.slider("Petal Lengh", 0,10)
pw = st.sidebar.slider("Petal Width", 0,10)

test = pd.DataFrame([[sl,sw,pl,pw]])

prediction = model.predict(test)
predicted_species = target_name[prediction[0]]

st.write("My prediction is ",predicted_species)