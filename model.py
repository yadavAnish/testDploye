import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 
plt.style.use('seaborn-bright')

from pandas import DataFrame

import sklearn
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from scipy import stats
from scipy.stats import norm, skew
import statsmodels.formula.api as sm

import warnings
import re
warnings.filterwarnings('ignore')

import pickle

# get_ipython().run_line_magic('matplotlib', 'inline')

#importdata
data = pd.read_csv('heart.csv')

print("Dataset : \n", data.head())
print("")
print ("Dataset Shape : ", data.shape)
print("")
data.info()
print("")
    
#check for null values
print("Check for null values : \n", data.isnull().sum())
print("")
    
    #check for unique values of the attributes
    #print("unique values of gender :", data.sex.unique())
    #print("unique values of chest pain type :", data.cp.unique())
    #print("unique values of fasting blood sugar > 120 mg/dl :", data.fbs.unique())
    #print("unique values of resting electrocardiographic results :", data.restecg.unique())
    #print("unique values of exercise induced angina :", data.exng.unique())
    #print("unique values of slp :", data.slp.unique())
    #print("unique values of number of major vessels :", data.caa.unique())
    #print("unique values of thall :", data.thall.unique())
    #print("")
    


#pairwise correlation
print("Pairwise correlation : \n", data.corr())
print("")

#split dataset in to train data and test data

# Separating the target variable
X,Y = data[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
            'exng', 'oldpeak', 'slp', 'caa', 'thall']], data[['output']]
    
# Splitting the dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.33, random_state=42)

#train model using gini index

tree = DecisionTreeClassifier(criterion = "gini", random_state = 35, max_depth=3, min_samples_leaf=2, 
                                  splitter="random", min_samples_split=5, max_features=None, max_leaf_nodes=None)
tree.fit(X_train, Y_train)

#model reuslts prediction
predicted = tree.predict(X_test)

#calculate accuracy and show confusion matrix + accuracy report

predicted=predicted.reshape(100,1)
    
#accracy
error = abs(predicted - Y_test)
errorMean = round(np.mean(error),2)
#ac = 1-round(np.mean(error),2)
print("error mean : ", errorMean)
print("")
    
#confusion matrix
cm = confusion_matrix(Y_test, predicted)
print("confusion matrix : \n", cm)

#plot confusion matrix
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Wistia)
classNames = ['Negative', 'Positive']
plt.title('Confusion Matrix - Test Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation = 45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN','TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+ " = "+str(cm[i][j]))
plt.show()
    
#prediction probability
predicted_prob = tree.predict_proba(X_test)[:,1]
    
# Accuray report AUC
accuracy = metrics.accuracy_score(Y_test, predicted)
auc = metrics.roc_auc_score(Y_test, predicted_prob)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc,2))

# Precision e Recall
recall = metrics.recall_score(Y_test, predicted)
precision = metrics.precision_score(Y_test, predicted)
F1_score = metrics.f1_score(Y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("F1 score :", round(F1_score,2))
print("Detail:")
print(metrics.classification_report(Y_test, predicted, target_names=[str(i) for i in np.unique(Y_test)]))

#generate the pickel file
pickle.dump(tree, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

    
