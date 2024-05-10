# Ex.9 Implementation-of-SVM-For-Spam-Mail-Detection

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
import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```
<br>
<br>
<br>
<br>
<br>  

## Output: 
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/0199a63b-138b-4c63-84b9-9d7ce4cbdbb8.png" width="700">   
<br>  

<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/10e5025a-a0dd-4e57-8108-733c7ebbcdaa.png" width="120">  
<br>  

<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/b7082818-4faa-4ec0-bc3a-63e89246128a.png" width="120">  
<br>  

<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/6c1b2631-c355-41dd-a8a4-4c3372237636.png" width="120">  
<br>  

<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/8d16cc79-4981-4f04-9d82-b3f7ec44b9ac.png">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/c176868c-fb37-4107-8b80-da8567adc015.png" width="310">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/6cf5747d-55df-4481-8de9-a07822c4eb5b.png" width="310">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/bfc4e24a-0a78-491f-9af7-348bcde11188.png" width="310">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/29bcfc1a-7b9e-4f73-8dd7-e32d869ada6f.png">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/d551c960-7a66-4738-a043-fe94b48fcf68.png">
<img src="https://github.com/DonBoscoBlaiseA/Implementation-of-SVM-For-Spam-Mail-Detection/assets/140850829/7d85fd71-191a-4771-93b2-95e9c5a1d41e.png">

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
