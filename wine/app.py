import streamlit as st 
import numpy as np 
from PIL import Image
import base64
import io

import os
import requests
import numpy as np
import pandas as pd


#st.title('Title')
st.header('ワイン品質予測モデル')
st.image("/app/streamlit/wine/images/wine.jpg", width=300)
st.write('[Databricksにおける機械学習モデル構築のエンドツーエンドのサンプル \- Qiita](https://qiita.com/taka_yayoi/items/f48ccd35e0452611d81b)') # markdown

# Copy and paste this code from the MLflow real-time inference UI. Make sure to save Bearer token from 
def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  token = os.environ.get("DATABRICKS_TOKEN")
  url = '<Model URL>'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  #st.write(token)
  
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

st.subheader('予測結果サンプル')

# sample data result
df = pd.read_csv("/app/streamlit/wine/data/sample_data.csv")
response = score_model(df) 
df['prediction'] = response
st.write(df)

st.subheader('テストデータのアップロード')
st.write('最初の1行目のみが処理されます。')

# data upload interface
csv_file_buffer_single = st.file_uploader('1レコードのみを含むCSVをアップロードしてください', type='csv')
if csv_file_buffer_single is not None:
  df = pd.read_csv(csv_file_buffer_single)
  #st.write(df)

  response = score_model(df[:1]) 
  df['prediction'] = response
  st.write(df)
  
  probability = int(df['prediction'][0] * 100)
  st.metric(label="このワインが高品質である確率", value=f"{probability}%")
