import pickle
import streamlit as st 
import pandas as pd 

df = pd.read_csv('train.csv')

try:
    model = pickle.load(open('modelusersurvive.pk','rb'))
except Exception as e:
    st.error(f'error is {e}')

st.title('model for survive prediction')

Pclass1 = sorted(df['Pclass'].unique())
Age1 = sorted(df['Age'].unique())
SibSp1 = sorted(df['SibSp'].unique())
Parch1 = sorted(df['Parch'].unique())
Fare1 = sorted(df['Fare'].unique())
Sex1 = sorted(df['Sex'].unique())
Embarked1 = sorted(df['Embarked'].dropna().unique())

Pclass = st.selectbox('Pclass1',Pclass1)
Age = st.selectbox('Age1',Age1)
SibSp = st.selectbox('sibSp1',SibSp1)
Parch = st.selectbox('Parch1',Parch1)
Fare = st.selectbox('Fare1',Fare1)
Sex = st.selectbox('Sex1',Sex1)
Embarked = st.selectbox('Embarked1',Embarked1)


Pclass = st.number_input('Pclass')
Age = st.number_input('Age')
SibSp = st.number_input('sibSp')
Parch = st.number_input('Parch')
Fare = st.number_input('Fare')
Sex = st.number_input('Sex')
Embarked = st.number_input('Embarked')


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
        st.success("Survived ")
    else:
        st.success("Not survived")

