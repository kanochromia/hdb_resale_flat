import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import random
from sklearn.preprocessing import StandardScaler

# import data file csv
df = pd.read_pickle('df_dum.pkl')
# set page title
st.set_page_config('HDB Resale Flat Prices')

st.title('Predict HDB Flat Prices (in Singapore Dollars)')
social_acc = ['About', 'LinkedIn']
social_acc_nav = st.sidebar.selectbox('About', social_acc)
if social_acc_nav == 'About':
    st.sidebar.markdown("<h2 style='text-align: center;'> Nazira Rasol</h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''
    • Data Science student in General Assembly Singapore''')

elif social_acc_nav == 'LinkedIn':
    st.sidebar.image('linkedin.jpg')
    st.sidebar.markdown("[Visit LinkedIn account](https://www.linkedin.com/in/nazira-rasol)")
    
menu_list = ['Exploratory Data Analysis', "Predict Price"]
menu = st.radio("Menu", menu_list)

if menu == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis of Resale Flat Prices')

    if st.checkbox("View data"):
        st.write(df)
        
        # more info about the EDA
        
elif menu == 'Predict Price':
    
    # dictionary
    flat_type_dic = {'type s1':0, 'improved':1, 'new generation':2, 'standard':3, 'model a':4, 'simplified':5,
                 'premium apartment':6, 'maisonette':7, 'model a-maisonette':8, 'apartment':9,
                 'adjoined flat':10, 'multi generation':11, 'model a2':12, 'terrace':13,
                 'improved-maisonette':14, 'premium maisonette':15, 'dbss':16, '2-room':17,
                 'type s2':18, 'premium apartment loft':19}
    flat_model_dic = {'2 room':0, '3 room':1, '4 room':2, '5 room':3, 'executive':4, 'multi-generation':5, '1 room':6}
    
    
    # list
    flat_type_list = ['type s1', 'improved', 'new generation', 'standard', 'model a', 'simplified',
                 'premium apartment', 'maisonette', 'model a-maisonette', 'apartment',
                 'adjoined flat', 'multi generation', 'model a2', 'terrace',
                 'improved-maisonette', 'premium maisonette', 'dbss', '2-room',
                 'type s2', 'premium apartment loft']
    flat_model_list = ['2 room', '3 room', '4 room', '5 room', 'executive', 'multi-generation', '1 room']
    
    # slider
    floor_square_area = st.slider("Enter preferred square footage", 31, 307)
    remaining_lease_years = st.number_input('Enter your preferred remaining lease, in years  (range = 43 - 100)')
    
    # select choices
    flat_type_choice = st.selectbox(label='Select your preferred flat type', options=flat_type_list)
    flat_types = flat_type_dic[flat_type_choice]
    
    flat_model_choice = st.selectbox(label='Select your preferred flat model', options=flat_model_list)
    flat_models = flat_model_dic[flat_model_choice]
    
    X = df.drop(columns=['resale_price'])
    y = df['resale_price']
    
    # Train, Test, Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scaled
    sc = StandardScaler()
    Z_train = sc.fit_transform(X_train)
    Z_test = sc.transform(X_test)
    
    # instantiate
    rf = RandomForestRegressor(n_estimators=10)
    line = LinearRegression()
    
    # Fit
    line.fit(Z_train, y_train)
    rf.fit(Z_train, y_train)
    
    # Score
    rf_score = rfc.score(Z_test, y_test)
    line_score = line.score(Z_tet, y_test)
    column_data = X.columns.values
    
    # if user selects randomforest
    
    def predict_price_rf(flattype, flatmodel, remaininglease, floorsqm):
        try:
            flat_type_index = flat_type_list.index(flattype)[0][0]
            flat_model_index = flat_model_list.index(flatmodel)[0][0]
        except ValueError:
            flat_type_index = -1
            flat_model_index = -1

        x = np.zeros(len(column_data))
        x[0] = remaininglease
        x[1] = floorsqm
        if flat_type_index >= 0:
            x[flat_type_index] = 1
        elif flat_model_index >= 0:
            x[flat_model_index] = 5


        return rf.predict([x])[0]
    
    # if user selects linear regression
    
    def predict_price_line(flattype, flatmodel, remaininglease, floorsqm):
        try:
            flat_type_index = flat_type_list.index(flattype)[0][0]
            flat_model_index = flat_model_list.index(flatmodel)[0][0]
        except ValueError:
            flat_type_index = -1
            flat_model_index = -1

        x = np.zeros(len(column_data))
        x[0] = remaininglease
        x[1] = floorsqm
        if flat_type_index >= 0:
            x[flat_type_index] = 1
        elif flat_model_index >= 0:
            x[flat_model_index] = 5


        return line.predict([x])[0]
    
    # option for linear regression or random forest
    
    alg = ['Random Forest Regression', 'Linear Regression']
    select_alg = st.selectbox('Choose Algorithm for Efficient Predict', alg)
    if st.button('Predict'):
        if select_alg == 'Random Forest Regression':
            st.write('Accuracy Score', rf_score)
            st.subheader(predict_price_rf(flattype, flatmodel, remaininglease, floorsqm))
            st.markdown("<h5 style='text-align: left;'> SGD </h5>", unsafe_allow_html=True)

        elif select_alg == 'Linear Regression':
            st.write('Accuracy Score', line_score)
            predicted_price = st.subheader(predict_price_line(flattype, flatmodel, remaininglease, floorsqm))
            st.markdown("<h5 style='text-align: left;'> SGD </h5>", unsafe_allow_html=True)
            if predict_price_linear(models, year, engine_size, transmissions, fuels) <= 0:
                st.write('Curious about why Linear Regression received Negative value as a Prediction. Here are '
                         'some resources which would make you understand mathematics behind Linear Regression better. ')
                st.markdown("[Stack Overflow answer](https://stackoverflow.com/questions/63757258/negative-accuracy-in-linear-regression)")
                st.markdown("[Quora](https://www.quora.com/What-is-a-negative-regression)")
                st.markdown("[Edureka Video on Linear regression ](https://www.youtube.com/watch?v=E5RjzSK0fvY)")
                st.write('Hope this helps you!')

                st.markdown('---')

    
    