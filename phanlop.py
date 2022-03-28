
#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
print('Libraries Imported')

#Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('d:/data/python111.csv', header = None)

#Renaming the columns
dataset.columns = ['c1', 'c2','c3','c4','c5','c6','c7','c8']
print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head())

#Splitting the data into independent and dependent variables
X = dataset.iloc[:,0:7].values
y = dataset.iloc[:,7].values
print('The independent features set: ')
print(X[:8,:])
print('The dependent variable: ')
print(y[:8])

# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print('hiii')

print(classifier.predict(X_test))


# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))

#Độ chính xác khi phân lớp
from sklearn.metrics import accuracy_score
print("Accuracy test: ", accuracy_score(y_test, y_pred))

# Lưu mô hình
print(list(zip(dataset.columns[0:7], classifier.feature_importances_)))
joblib.dump(classifier, 'd:/data/randomforestmodel.pkl') 




