# -*- coding: utf-8 -*-

#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yukleme
veriler = pd.read_csv('D:\\ders\\ml_works\\veriler4.csv')


#Kategorik - numerik yapma odasayısı kolonu
odasayisi = veriler.iloc[:,0:1].values
print(odasayisi)

from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()
odasayisi[:,0] = labelencoder.fit_transform(veriler.iloc[:,0])
print(odasayisi)
ohe = preprocessing.OneHotEncoder()
odasayisi = ohe.fit_transform(odasayisi).toarray()
print(odasayisi)

#Kategorik - numerik ypma kimden kolonu
kimden = veriler.iloc[:,-1:].values
print(kimden)
from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()
kimden[:,-1] = labelencoder.fit_transform(veriler.iloc[:,-3])
print(kimden)
ohe = preprocessing.OneHotEncoder()
kimden = ohe.fit_transform(kimden).toarray()
print(kimden)


#numerik kolonlar
numerik = veriler.iloc[:,1:9].values
print(numerik)




#df
sonuc1 = pd.DataFrame(data=odasayisi, index = range(102), columns = ['2+1','3+1','4+1'])
print(sonuc1)

sonuc2 = pd.DataFrame(data=numerik, index = range(102), columns = ['m² (Brüt)','m² (Net)','Bina Yaşı','Bulunduğu Kat','Kat Sayısı','Banyo Sayısı','Aidat','Fiyat'])
print(sonuc2)


sonuc3 = pd.DataFrame(data = kimden, index = range(102), columns = ['Sahibinden','Emlak Ofisinden'])
print(sonuc3)




#dataframe birleştirme
s1=pd.concat([sonuc1,sonuc2], axis=1)
print(s1)

s2=pd.concat([s1,sonuc3], axis=1)
print(s2)



#verilerin ayrılması 

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s1,sonuc3,test_size=0.33, random_state=0)

#çoklu doğrusal regresyon işlemi
from sklearn.linear_model import LinearRegression


Fiyat = s2.iloc[:,10].values

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,Fiyat,test_size=0.33, random_state=0)


regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#model
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_pred)

#MSE, RMSE, RSQ değeri hesap
from sklearn.metrics import mean_squared_error
print("Ortalama Karesel Hata (MSE):")
print( mean_squared_error(y_test,y_pred)  )
print("Kök Ortalama Kare Hata (RMSE):")
print( mean_squared_error(y_test,y_pred, squared=False)  )
print("Korelasyon katsayısı (RSQ):")
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred) )


