import pickle
import streamlit as st 
import pandas as pd 
# loading the saved models 

model = pickle.load(open('modelusersurvive.pk1','rb'))
# creating the function for web app

st.title('model for survive prediction')

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

# if st.button('Predict'):
#     input_data = pd.DataFrame([{
#         'Pclass': Pclass,
#         'Age': Age,
#         'SibSp': SibSp,
#         'Parch': Parch,
#         'Fare': Fare,
#         'Sex': Sex,
#         'Embarked': Embarked
#     }])

#     # 👉 Encoding (example)
#     input_data = pd.get_dummies(input_data)

#     # 👉 Align with training columns
#     input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

#     # 👉 Scaling (if used)
#     input_data = model.transform(input_data)

#     # 👉 Prediction
#     prediction = model.predict(input_data)

#     if prediction[0] == 1:
#         st.success("Survived ✅")
#     else:
#         st.error("Not Survived ❌")