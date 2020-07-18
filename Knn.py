import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing
import pickle
data=pd.read_csv("car.data")
print(data.head())
le=preprocessing.LabelEncoder()
buying=le.fit_transform(list(data["buying"]))
maint=le.fit_transform(list(data["maint"]))
doors=le.fit_transform(list(data["doors"]))
persons=le.fit_transform(list(data["persons"]))
lug_boot=le.fit_transform(list(data["lug_boot"]))
safety=le.fit_transform(list(data["safety"]))
cls=le.fit_transform(list(data["class"]))
x=list(zip(buying,maint,doors,persons,lug_boot,safety))
y=list(cls)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)
best=0.7
'''for i in range(40):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train,y_train)
    acc=model.score(x_test,y_test)
    print(acc)
    if(acc>best):
        best=acc
        with open("studentmodel2.pickle", "wb")as f:
            pickle.dump(model, f)'''
pickle_in=open("studentmodel2.pickle","rb")
model=pickle.load(pickle_in)
acc=model.score(x_test,y_test)
print(acc)
predicted=model.predict(x_test)
for x in range(len(predicted)):
    print("predicted: ",predicted[x],"data: ",x_train[x]," actual ",y_test[x])