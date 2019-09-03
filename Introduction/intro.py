from sklearn.tree import DecisionTreeClassifier #Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier
from sklearn.naive_bayes import GaussianNB #Naive Bayes Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score # For performance analysis

###  DATASET ###

#[height,weight,shoe size]
X = [[181,90,44],[172,87,42],[195,100,46],[170,55,41],[166,65,40],[175,64,39],[172,65,43],[189,78,44],[168,55,38]]
# List of labels
Y = ['male','female','male','female','female','female','male','male','female']
# Testing Values
test = [[190,70,43],[168,55,39],[170,95,42],[199,88,44],[188,60,42],[190,59,40],[165,70,42],[181,90,40],[175,74,41]]

### CLASSIFIERS ###

clf_tree =  DecisionTreeClassifier() #create Decision Tree classifier object
clf_tree = clf_tree.fit(X,Y) #train the Decision Tree classifier
clf_rf = RandomForestClassifier() #create Random Forest classifier object
clf_rf = clf_rf.fit(X,Y) #train the Random Forest classifier
clf_gnb = GaussianNB() #create Naive Bayes classifier
clf_gnb = clf_gnb.fit(X,Y) #train Naive Bayes classifier
clf_neigh = KNeighborsClassifier() #create KNeighbor Classifier
clf_neigh = clf_neigh.fit(X,Y) #train KNeighbor Classifier


### PREDICTIONS ### - use the different classifiers to predict the test array

prediction_tree = clf_tree.predict(test)
prediction_rf = clf_rf.predict(test)
prediction_gnb = clf_gnb.predict(test)
prediction_neigh = clf_neigh.predict(test)

### ACCURACY ### - find accuracy score of different classifiers

accuracy_tree= accuracy_score(Y,prediction_tree)*100
accuracy_rf = accuracy_score(Y,prediction_rf)*100
accuracy_gnb = accuracy_score(Y,prediction_gnb)*100
accuracy_neigh = accuracy_score(Y,prediction_neigh)*100

### RESULTS ### - print results and accuracy

print('Prediction Decision Tree Classifier:',prediction_tree,'Accuracy:',accuracy_tree)
print('Prediction Random Forest Classifier:',prediction_rf,'Accuracy:',accuracy_rf)
print('Prediction Naive Bayes Classifier:',prediction_gnb,'Accuracy:',accuracy_gnb)
print('Prediction KNeighbors Classifier:',prediction_neigh,'Accuracy:',accuracy_neigh)
