import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle

st.title('Unsupervise ML Application')

model = pickle.load(open('unmodel.pkl','rb'))
scaler = pickle.load(open('unscaler.pkl','rb'))



with st.form('my-form'):
    cgpa = st.number_input('cgpa')
    iq = st.number_input('iq')

    submit = st.form_submit_button('Submit')

    if submit:
        data = np.array([[cgpa,iq]])

        input_data = scaler.transform(data)
        prediction = model.predict(input_data)
        st.write('cluster',prediction[0])

    if prediction[0] == 0:
        st.write('sorry! You could not placed')
    elif prediction[0] == 1 or prediction[0] == 2 or prediction[0] ==3 or prediction[0] ==4:
        st.write('placed')
    elif prediction[0] ==5:
        st.write('60 to 70 percent chance to place')
    elif prediction[0] ==6:
        st.write('may be')
    elif prediction[0] ==7 or prediction[0] ==8 or prediction[0]==9:
        st.write('out of placement criteria')
    else:
        st.write('Try Next time')

centers = scaler.inverse_transform(model.cluster_centers_)

fig, ax = plt.subplots()
ax.scatter(centers[:,0], centers[:,1], marker='o', s=200)
st.pyplot(fig)

