import pickle
import streamlit as st 
import pandas as pd 

df = pd.read_csv('train.csv')

try:
    model = pickle.load(open('modelusersurvive.pk','rb'))
except Exception as e:
    st.error(f'error is {e}')

st.title('model for survive prediction')

P_class = sorted(df['Pclass'].unique())
P_Age = sorted(df['Age'].unique())
P_SibSp = sorted(df['SibSp'].unique())
P_Parch = sorted(df['Parch'].unique())
P_Fare = sorted(df['Fare'].unique())
P_Sex = sorted(df['Sex'].unique())
p_Embarked = sorted(df['Embarked'].dropna().unique())

Pclass = st.selectbox('P_class',P_class)
Age = st.selectbox('P_Age',P_Age)
SibSp = st.selectbox('P_SibSp',P_SibSp)
Parch = st.selectbox('P_Parch',P_Parch)
Fare = st.selectbox('P_Fare',P_Fare)
Sex = st.selectbox('P_Sex',P_Sex)
Embarked = st.selectbox('P_Embarked',p_Embarked)



if st.button('predict'):
    input_data = pd.DataFrame([{
        'Pclass':Pclass,
        'Age':Age,
        'SibSp':SibSp,
        'Parch':Parch,
        'Fare':Fare,
        'Sex':Sex,
        'Embarked':Embarked
    }])
    predections = model.predict(input_data)
    st.success(f'prediction is : {predections[0]}')
    if predections[0] == 1:
        st.success("passanger can survive ")
    else:
        st.success(" passanger can not survive")

