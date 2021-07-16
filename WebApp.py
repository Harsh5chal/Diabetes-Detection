
#Import the required libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

#Create a title and sub title
st.write(
    """
    # Diabetes Detection for Indians
    Detects if someone has diabetes in india using Machine Learning and Python !
    """
)

#Import the diabetes dataset
df = pd.read_csv('https://raw.githubusercontent.com/Harsh5chal/Diabetes-Detection/main/pima-indians-diabetes.data', header=None)

#Set a subheader
st.subheader('Data Information:')

#Show the data as a table
st.dataframe(df)

#Show Stats
st.write(df.describe())

#Show data as chart
chart = st.bar_chart(df)

#Split the data into independent X and dependent Y
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#Split the dataset 75% - 25%
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

#Get the feature input from the user
def get_user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    Blood_Pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    Skin_Thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)
    
    #Store a dictonary into a variable
    user_data = { 
    'Pregnancies' : Pregnancies,
    'Glucose' : Glucose,
    'Blood Pressure' : Blood_Pressure,
    'Skin Thickness' : Skin_Thickness,
    'Insulin' : Insulin,
    'BMI' : BMI,
    'DPF' : DPF,
    'Age' : Age
    }
    
    #Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#Store the user input into a variable
user_input = get_user_input()

#Set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show Model metrices
st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%' )

#Store Model Prediction
prediction = RandomForestClassifier.predict(user_input)

#Set a subheader display the classicfication
st.subheader('Classification:')
st.write(prediction)
