import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Breast_cancer_data.csv")
print (df)
X = df.drop(["diagnosis"], axis = 1) # cột diagnosis
y = df["diagnosis"] # lay gia tri cot diagnos
print (X)
print (y)

#train-test_split : chia mang hoac ma tran thanh tap con thu nghiem va huan luyen
from sklearn.model_selection import train_test_split
#test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

print("X_train là:")
print (X_train)
print("Y_train là:")
print (y_train)
print("X_test là:")
print (X_test)
print("y_test là:")
print (y_test)
print("X shape:",X.shape)
print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)

from sklearn.svm import SVC
log_model=SVC(kernel="linear").fit(X_train,y_train)
y_pred=log_model.predict(X_test)
print("Hệ số w:",log_model.coef_)
print(log_model.coef_.shape)
print("Hệ số bias:",log_model.intercept_)
print("Số lớp:",log_model.classes_)

#Đánh giá mô hình học dựa trên kết quả dự đoán (với độ đo đơn giản Accuracy, Precision, Recall)
from sklearn.metrics import accuracy_score
print("Accuracy Score:", accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix1 = confusion_matrix(y_test,y_pred)
print(confusion_matrix1)



X= df[['mean_texture']] 
y= df['diagnosis'] 
x0 = X[y==0]
x1 = X[y==1]

plt.plot(x0['mean_texture'],'b^', markersize = 4, alpha = .8)
plt.plot(x1['mean_texture'],'go', markersize = 4, alpha = .8)
plt.xlabel('')
plt.ylabel('mean_texture')
plt.plot()
plt.show()


 
