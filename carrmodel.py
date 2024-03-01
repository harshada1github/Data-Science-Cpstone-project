import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import Lasso
from sklearn import metrics   
df = pd.read_csv('CAR DETAILS.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
df.duplicated().sum()
df.drop_duplicates(keep='first') 
df.shape
df.head()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['fuel'].value_counts()
df['fuel'] = encoder.fit_transform(df['fuel'])
df['fuel'].value_counts()
df['seller_type'].value_counts()
df['seller_type'] = encoder.fit_transform(df['seller_type'])
df['seller_type'].value_counts()
df['transmission'].value_counts()
df['transmission'] = encoder.fit_transform(df['transmission'])
df['transmission'].value_counts()
df['owner'].value_counts()
df['owner'] = encoder.fit_transform(df['owner'])
df['owner'].value_counts()
df.head()
df.to_csv('sample_data.csv')
x = df.drop(['selling_price','name'] , axis = 1)
y = df['selling_price']
x.shape ,y.shape
from sklearn.model_selection import train_test_split 
x_train ,x_test ,y_train ,y_test = train_test_split(x,y,test_size = 0.20 ,random_state = 40)
x_train.shape , x_test.shape , y_train.shape , y_test.shape
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train , y_train)
regression.fit(x_train , y_train)
y_pred_test = regression.predict(x_test)
y_pred_test
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred_test)
print(score)
from sklearn.linear_model import Lasso
lass_reg_model = Lasso()
lass_reg_model.fit(x_train,y_train)
training_data_prediction = lass_reg_model.predict(x_train)
error_score = metrics.r2_score(y_train, training_data_prediction)
print("R squared Error : ", error_score)
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(x_train , y_train)
y_pred = rf_reg.predict(x_test)
rf_reg.score(x_train , y_train)
rf_reg.score(x_test , y_test)
import pickle
pickle.dump(rf_reg,open('random_regressor.pkl','wb'))
model =pickle.load(open('random_regressor.pkl','rb'))
model.predict(x_test)

