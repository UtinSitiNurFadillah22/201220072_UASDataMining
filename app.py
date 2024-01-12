import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

# Mengimport model 
dc = pickle.load(open('DC.pkl','rb'))

# Menload dataset
data = pd.read_excel('datasetharga.xlsx')


st.title('Aplikasi Price Handphone')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Costumer Churn</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1, unsafe_allow_html=True)
activities = [' ', 'DecisionTreeClassifier']
option = st.sidebar.selectbox('Model Name',activities)
st.sidebar.header('Data Handphone')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p> Dataset Price Range Phone merupakan dataset yang menampilkan harga handphone dan kecocokan pengguna</p>
    """
    st.markdown(html_layout2, unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe Dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

# Proses training test split
X = data.drop('price_range',axis=1)
y = data['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)

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
    # Input untuk fitur numerik
    battery_power = st.sidebar.slider('battery_power', 0, 1000, 4000)
    clock_speed = st.sidebar.slider('clock_speed', 0.1, 0.5, 2.0)
    dual_sim = st.sidebar.radio('dual_sim', ['Yes', 'No'])
    fc = st.sidebar.slider('fc', 0, 80, 25)
    four_g = st.sidebar.radio('four_g', ['Yes', 'No'])
    int_memory = st.sidebar.slider('int_memory', 2, 256, 16)
    m_dep = st.sidebar.slider('m_dep', 0.0, 1.0, 0.5)
    mobile_wt = st.sidebar.slider('mobile_wt', 80, 200, 150)
    n_cores = st.sidebar.slider('n_cores', 1, 8, 4)
    pc = st.sidebar.slider('pc', 0, 10, 5)
    px_height = st.sidebar.slider('px_height', 0, 2000, 500)
    px_width = st.sidebar.slider('px_width', 0, 2000, 1000)
    ram = st.sidebar.slider('ram', 512, 8192, 2048)
    sc_h = st.sidebar.slider('sc_h', 5, 30, 15)
    sc_w = st.sidebar.slider('sc_w', 3, 20, 10)
    talk_time = st.sidebar.slider('talk_time', 2, 24, 12)
    three_g = st.sidebar.radio('three_g', ['Yes', 'No'])
    touch_screen = st.sidebar.radio('touch_screen', ['Yes', 'No'])
    wifi = st.sidebar.radio('wifi', ['Yes', 'No'])
    price_range = st.sidebar.selectbox('price_range', [1, 2, 3, 4])

    # Mengonversi kategori 'Yes'/'No' ke nilai float
    dual_sim = 0.0 if dual_sim == 'No' else 1.0
    four_g = 0.0 if four_g == 'No' else 1.0
    three_g = 0.0 if three_g == 'No' else 1.0
    touch_screen = 0.0 if touch_screen == 'No' else 1.0
    wifi = 0.0 if wifi == 'No' else 1.0

    # Membuat DataFrame
    user_report_data = {
        'battery_power': battery_power,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'fc': fc,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': three_g,
        'touch_screen': touch_screen,
        'wifi': wifi,
        'price_range': price_range
    }
        
    report_data = pd.DataFrame(user_report_data, index=[0])

    return report_data


#Data Pasion
user_data = user_report()
st.subheader('Data Price Phone')
st.write(user_data)

user_result = dc.predict(user_data)
nb_score = accuracy_score(y_test,dc.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Handphone ini Cocok Untuk Kamu'
else:
    output ='Handphone ini Tidak Cocok Untuk Kamu'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(nb_score*100)+'%')