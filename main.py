import pandas as pd
import numpy as np



# import data_set
data = pd.read_csv("breast.csv")
data.dropna(inplace=True) # for removing null values


# splitting to independent and target values
x = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# split to training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
#print(x_train, y_train)

# no need for feature scaling because we are using decision tree classification algorithm
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x_train = st.fit_transform(x_train)
x_test = st.transform(x_test)

# visualization
import matplotlib.pyplot as plt
# first we present the correlation matrix
print(data.iloc[:, 2:].corr())
plt.scatter(x_train[:, 0], x_train[:, 2], color='red') # radius_mean and perimeter_mean
plt.scatter(x_train[:, 0], x_train[:, 3], color='blue') # radius_mean and area_mean
plt.xlabel("radius_mean")
plt.ylabel("periemeter_mean and area_mean")
plt.show()


# learning (fitting the model)

# fitting the decision tree model
from sklearn.tree import DecisionTreeClassifier
DTclassifier = DecisionTreeClassifier(criterion='gini', random_state=10)
DTclassifier.fit(x_train, y_train)

# fitting the knn model
from sklearn.neighbors import KNeighborsClassifier
KNNclassifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
KNNclassifier.fit(x_train, y_train) # lazy learner

# fitting the Naive Bays model
from sklearn.naive_bayes import GaussianNB
NBclassifier = GaussianNB()
NBclassifier.fit(x_train, y_train)

# prediction on test set
y_pred1 = DTclassifier.predict(x_test)
y_pred2 = KNNclassifier.predict(x_test)
y_pred3 = NBclassifier.predict(x_test)

# measuring the accuracy of the model
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)

print(cm1)
print(DTclassifier.score(x_test, y_test))
print(cm2)
print(KNNclassifier.score(x_test, y_test))
print(cm3)
print(NBclassifier.score(x_test, y_test))

test_row = [[7.76,24.54,47.92,181,0.05263,0.04362,0,0,0.1587,0.05884,0.3857,1.428,2.548,19.15,0.007189,0.00466,0,0,0.02676,0.002783,9.456,30.37,59.16,268.6,0.08996,0.06444,0,0,0.2871,0.07039]]
test_row = st.transform(test_row)
predict = KNNclassifier.predict(test_row)
print(predict)

