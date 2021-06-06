import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")
print (df.head())
print("Dữ liệu đầu vào là:")
X = df.drop(["target","cp","fbs","restecg","exang","slope","ca","thal"], axis = 1)
print (X)
print("Dữ liệu đầu ra là:")
y = df["target"]
print (y)

# chuẩn hóa
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)
print("Chuẩn hóa dữ liệu:\n")
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

print("X_train là:")
print (X_train)
print("Y_train là:")
print (y_train)
print("X_test là:")
print (X_test)
print("y_test là:")
print (y_test)


print("Mô hình SVM:")
from sklearn.svm import SVC
log_model=SVC(kernel="linear").fit(X_train,y_train)

y_pred=log_model.predict(X_test)
print("\n Tập y dự đoán:")
print(y_pred)

print("Hệ số w:\n",log_model.coef_)
print(log_model.coef_.shape)
print("Hệ số bias:",log_model.intercept_)
print("Số lớp:",log_model.classes_)

from sklearn.metrics import accuracy_score
print("Tỷ lệ:",accuracy_score(y_test,y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix1 = confusion_matrix(y_test,y_pred)
print(confusion_matrix1)


X= df[['trestbps']] 
y= df['target'] 
x0 = X[y==0]
x1 = X[y==1]

plt.plot(x0['trestbps'],'b^', markersize = 4, alpha = .8)
plt.plot(x1['trestbps'],'go', markersize = 4, alpha = .8)
plt.xlabel('')
plt.ylabel('trestbps')
plt.plot()
plt.show()


 
