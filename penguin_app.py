# Importing the necessary libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

@st.cache(suppress_st_warning=True)
def prediction(model,island,b_length,b_depth,f_length,mass,sex):
  arr_input=[[island,b_length,b_depth,f_length,mass,sex]]
  if model_str=='Support Vector Machine':
    y_pred=svc_model.predict(arr_input)
    score = svc_model.score(X_train, y_train)
  elif model_str=='Logistic Regression':
    y_pred=log_reg.predict(arr_input)
    score = log_reg.score(X_train, y_train)
  elif model_str=='Random Forest Classifier':
    y_pred=rf_clf.predict(arr_input)
    score = rf_clf.score(X_train, y_train)
  y_val=y_pred[0]
  if y_val==0:
    return ['Adelie',score]
  elif y_val==1:
    return ['Chinstrap',score]
  elif y_val==2:
    return ['Gentoo',score]
st.sidebar.title("Penguin Species Prediction App")
model_str=st.sidebar.selectbox('Classifier Model', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
island_str=st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
sex_str=st.sidebar.selectbox('Sex', ('Male', 'Female'))
b_length=st.sidebar.slider("Bill Length",float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))
b_depth=st.sidebar.slider("Bill Depth",float(df['bill_depth_mm'].min()),float(df['bill_depth_mm'].max()))

f_length=st.sidebar.slider("Flipper Length",float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()))

mass=st.sidebar.slider("Body Mass",float(df['body_mass_g'].min()),float(df['body_mass_g'].max()))
if sex_str=='Male':
  sex=0
elif sex_str=='Female':
  sex=1
if island_str=='Biscoe':
  island=0
elif island_str=='Dream':
  island=1
elif island_str=='Torgersen':
  island=2
if st.sidebar.button("Predict"):
  x=prediction(model_str,island,b_length,b_depth,f_length,mass,sex)
  st.write("Species predicted: ", x[0])
  st.write("Accuracy score of this model is: ", x[1])
