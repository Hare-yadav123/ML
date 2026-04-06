import pandas as pd
import numpy as np 
import pickle 
import streamlit as st 

model = pickle.load(open('IPLModel.pkl','rb'))
le = pickle.load(open('labelencoder.pkl','rb'))
df = pd.read_csv('matches.csv')

df['Season'] = df['Season'].astype(str).str.replace('IPL-', '', regex=False)
df['Season'] = df['Season'].str.extract(r'(\d+)').astype(int)

st.title('IPL match winner precdictor')

season_list = sorted(df['Season'].unique())

city_list = sorted(df['city'].dropna().unique())
team1_list = sorted(df['team1'].unique())
team2_list = sorted(df['team2'].unique())
toss_winner_list = sorted(df['toss_winner'].unique())
toss_decision_list = sorted(df['toss_decision'].unique())
venue_list = sorted(df['venue'].unique())


Season = st.selectbox('Season',season_list)
city = st.selectbox('city',city_list)
team1 = st.selectbox('team1',team1_list)
team2 = st.selectbox('team2',team1_list)
toss_winner = st.selectbox('toss_winner',toss_winner_list)
toss_decision = st.selectbox('toss_decision',toss_decision_list)
venue = st.selectbox('venue',venue_list)

if  st.button('Predict'):
    input_data = pd.DataFrame([{
        'Season':Season,
        'city':city,
        'team1':team1,
        'team2':team2,
        'toss_winner':toss_winner,
        'toss_decision':toss_decision,
        'venue' : venue
    }])

    prediction = model.predict(input_data)
    finaloutput = le.inverse_transform(prediction)
    st.write(f"The winner of this match is : {finaloutput[0]}")