from django.shortcuts import render
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score


def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pickle_file = "C:/Users/HP/Desktop/TEST DataAnalysis/model.pk2_0801"
    with open(pickle_file, 'rb') as f:
        model = pickle.load(f)
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    
    result1 = ""
    if pred == [1]:
        result1 = "   Patient is Likely Positive"
    else:
        result1 = "   Patient is Likely Negative"



    return render(request, 'predict.html', {'result1':result1})
