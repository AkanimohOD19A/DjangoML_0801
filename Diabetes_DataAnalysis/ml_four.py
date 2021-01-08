#import the libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', DeprecationWarning)

df = pd.read_csv("C:/Users/HP/Desktop/TEST DataAnalysis/Test0801/Diabetes_DataAnalysis/diabetes.csv")



X = df.drop('Outcome', 1)
y = df.iloc[:, -1]

def compute_score(clf, X, y, scoring = 'accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring = scoring)
    return np.mean(xval) # Cross validation to check for biases and variance

#Testing different base models
logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()
models = [logreg, logreg_cv, rf, gboost]


for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=X, y=y, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')


#Lets try train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


#Using the random forest algorithm
model = rf.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("##############################################")
print("Testing Model Prediction")
pred = model.predict([[6,148,72,35,0,33.6,0.627,50]])
if pred == [1]:
    print("Positive")
else:
    print("Negative")

print(pred, '\n')
print("END OF TEST")
print("##############################################")

#Check the prediction precision and accuracy
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

#Saving the model with pickle
import pickle

# save the model to disk
model_name  = 'model.pk2_0801'
pickle.dump(model, open(model_name, 'wb'))

print("[INFO]: Finished saving model...")