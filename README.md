# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Don Bosco Blaise A
RegisterNumber: 212221040045
*/
```
```
import chardet
file = 'spam.csv'
with open(file,'rb') as rawdata:
  result= chardet.detect(rawdata.read(100000))
result

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Spam.csv',encoding='latin-1')
df = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df.head()

df.info()

df.isnull().sum()

x=df["v1"].values
y=df["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```  

## Output: 
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/0412ef25-3fe5-42ae-b4c5-9d5d878bc408.png" width="600"> horizontal
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/9382c97e-0dd1-4ffc-9009-90ff0874bd9d.png" width="600"> horizontal
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/0dfa3686-a47d-4613-8425-a20cc883e426.png" width="600"> horizontal
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/ca11655f-4a93-432f-b214-bcaacce384b9.png" width="600"> horizontal
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/d4b4ec8a-d1d6-45d5-b605-10ed1b4b08db.png" width="600"> horizontal

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
