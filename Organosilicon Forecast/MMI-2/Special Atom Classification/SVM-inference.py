import pickle
import pandas as pd


# 加载数据
data = pd.read_excel('your_file_name.xlsx')

f2=open('svm.model','rb')
s2=f2.read()
model1=pickle.loads(s2)
expected = data
predicted = model1.predict(data)