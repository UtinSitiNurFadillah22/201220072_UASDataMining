import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns 
import pickle 
from sklearn.tree import DecisionTreeClassifier

#import model 
svm = pickle.load(open('Price Range Phone Dataset.csv', 'rb'))

#load dataset
data = pd.read_csv('Price Range Phone Dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Mobile Price Classification')

html_layout1 = """
<br>
<div style="background-color:blue ; padding:5px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Mobile Phone Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Decision Tree']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data spesifikasi')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Price Range Phone</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

#train test split
# Perbarui dataset
X_new = data[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc']]
y_new = data['price_range']
X = data.drop('price_range',axis=1)
y = data['price_range']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.20, random_state=42)
svm.fit(X_train, y_train)
# Simpan model baru
pickle.dump(svm, open('Updated.pkl', 'wb'))


#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)
    

def user_report():
    battery_power = st.sidebar.slider('battery_power',0,20,1)
    blue = st.sidebar.slider('blue',0,200,108)
    clock_speed = st.sidebar.slider('clock_speed',0,140,40)
    dual_sim = st.sidebar.slider('dual_sim',0,2)
    fc = st.sidebar.slider('fc',0,1000,120)
    
    user_report_data = {
        'battery_power':battery_power,
        'blue':blue,
        'clock_speed':clock_speed,
        'dual_sim':dual_sim,
        'fc':fc,
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data')
st.write(user_data)

user_result = svm.predict(user_data)
svc_score = accuracy_score(y_test,svm.predict(X_test))


models = [DecisionTreeClassifier]
model_names = ['Decision Tree']
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Hp Ini Direkomendasikan'
else:
    output ='Hp Ini Tidak Direkomendasikan'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(svc_score*100)+'%')