import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.cross_validation import train_test_split
from time import time
#  time python initial_gender_models.py >> initial_gender_models_output.txt

f1 = '../data/gender_age_train.csv'
f2 = '../data/phone_brand_device_model.csv'
df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)
df = df1.merge(df2, on='device_id')

le = preprocessing.LabelEncoder()
target = le.fit_transform(df['gender'].values)
features = pd.get_dummies(df['device_model'])
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.7)

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
t0 = time()
clf_fit = clf.fit(X_train, y_train)
t1 = time()
y_pred = clf_fit.predict(X_test)
y_prob = clf_fit.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred)
print 'Random Forrest Accuracy...'
print np.sum(np.diag(cm))*1. / np.sum(cm, axis=None)
print 'Confusion Matrix...'
print cm
print 'Log Loss...'
print log_loss(y_test, y_prob)
print 'Runtime...'
print str(t1 - t0)


