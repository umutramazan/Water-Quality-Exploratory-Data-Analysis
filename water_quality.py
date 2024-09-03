
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data
data=pd.read_csv("water_potability.csv")
#1->potable 0->non-potable 

print(data.head())
print(data.describe())
print(data.info())


correlation_matrix=data.corr()
print(correlation_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#hicbir ozellikle icilebilirligin arasında dogrudan bir iliski yoktur.
#bu nedenle her feature sonuca katkı saglayacaktır.

#ozelliklerin dagılımı

non_potable=data.query("Potability==0")
potable=data.query("Potability==1")

plt.figure(figsize=(10,8))

for ax,col in enumerate(data.columns[:9]):
    plt.subplot(3,3,ax+1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col],label="non-potable") 
    sns.kdeplot(x=potable[col],label="potable")
    plt.legend()
plt.tight_layout()
plt.show()

"""
egrileri olusturmak icin kdeplot kullanıyoruz.
ph degeri icilebilir ve icilemez de ortalama ve medyan degerleri neredeyse aynı oldugundan ayırt edici bir ozellik degildir
solid hafif sola yatik->pozitif carpıklık vardır.Yani medyan degerleri biraz farklı buna gore  icilebilir su ile icilemez sularda solid degerleri farklı
turbidity tamamen aynı sulfat tamamen farklı denilebilir.
"""

#eksik verilerin ortalama ile doldurulması
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

missing_data=data.iloc[:,[0,4,7]]

#imputeri secilen satırlara uygun hale getirir.
imputer=imputer.fit(missing_data)

#veriyi donustur ve dataFrame'i guncelle
data.iloc[:,[0,4,7]]=imputer.transform(missing_data)

#train test split ve normalizasyon
x=data.iloc[:,:9].values
y=data.iloc[:,9:].values


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=3)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X_train=scaler.fit_transform(x_train)
X_test=scaler.fit_transform(x_test)

"""
Test seti üzerinde fit yapılmaz çünkü bu işlem modelin test setinden bilgi almasına (data leakage) yol açabilir. 
Ve modelin performansını olduğundan daha iyi gösterir.
"""

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)

y_pred=dtc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("DTC")
print(cm)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("RF")
print(cm)

#sonuclar incelendiginde rf daha iyi sonuc verdi

#rf hyperparameter tuning (hiperparametre optimizasyonu)

from sklearn.model_selection import GridSearchCV

p=[{"n_estimators":[10,50,100],"max_features":["sqrt","log2"],"max_depth":list(range(1,21,3))}]

gs=GridSearchCV(estimator=rf,
                param_grid=p,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

grid_search=gs.fit(X_train,y_train)
best_result=grid_search.best_score_
best_parameters=grid_search.best_params_

print(best_result)
print(best_parameters)














