import pickle
import streamlit as st 
import pandas as pd 

df = pd.read_csv('train.csv')

try:
    model = pickle.load(open('modelusersurvive.pk','rb'))
except Exception as e:
    st.error(f'error is {e}')

st.title('model for survive prediction')

Pclass1 = sorted(df['Pclass1'].unique())
Age1 = sorted(df['Age1'].unique())
SibSp1 = sorted(df['SibSp1'].unique())
Parch1 = sorted(df['Parch1'].unique())
Fare1 = sorted(df['Fare1'].unique())
Sex1 = sorted(df['Sex1'].unique())
Embarked1 = sorted(df['Embarked1'].unique())

Pclass = st.selectbox('Pclass',Pclass1)
Age = st.selectbox('Age',Age1)
SibSp = st.selectbox('sibSp',SibSp1)
Parch = st.selectbox('Parch',Parch1)
Fare = st.selectbox('Fare',Fare1)
Sex = st.selectbox('Sex',Sex1)
Embarked = st.selectbox('Embarked',Embarked1)


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

