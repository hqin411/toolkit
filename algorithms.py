
#setup environment
import os 
import time
import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import datetime 
import json
import boto3
import s3fs
import pytz
import yaml
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras.models import model_from_json, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import os
import itertools
from utils import evaluate
import seaborn as sns

warnings.filterwarnings("ignore")

%load_ext autoreload
%autoreload 2

pd.options.display.max_columns = None

s3 = boto3.resource('s3')
s3c = boto3.client('s3')
s3sys = s3fs.S3FileSystem()

train_name = ''
val_name = ''
test_name = ''


#read parquet data from S3
train_data = pq.ParquetDataset(train_name, filesystem=s3sys).read_pandas().to_pandas()
val_data = pq.ParquetDataset(val_name, filesystem=s3sys).read_pandas().to_pandas()
test_data = pq.ParquetDataset(test_name, filesystem=s3sys).read_pandas().to_pandas()


#tuning
def train_and_evaluate_3_layers(n_unit,
                                dropout,
                                lr,
                                batch_size,
                                epoch,
                                training_data,
                                validation_data,
                                shape_size):
    inputs = Input(shape=(shape_size,))
    X = Dense(n_unit, activation='relu')(inputs)
    X = Dropout(dropout)(X)
    X = Dense(n_unit, activation='relu')(X)
    X = Dropout(dropout)(X)
    predictions = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=inputs, outputs=predictions)
    optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=1e-5, nesterov=True)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_data[0], training_data[1], batch_size=batch_size, epochs=epoch, validation_split=0.0, shuffle=True, validation_data=validation_data)
    y_probas = [r[0] for r in model.predict(validation_data[0])]
    auc = roc_auc_score(y_true=validation_data[1], y_score=y_probas)
    return auc

n_units = [32, 64, 128]
dropouts = [0.2, 0.4, 0.6]
lrs = [0.005, 0.01, 0.03]
batch_sizes = [32, 64, 128]
epochs = [5]
aucs = []

shape_size = pipeline.transform(train_data).shape[1]

for n_unit, dropout, lr, batch_size, epoch in itertools.product(n_units, dropouts, lrs, batch_sizes, epochs):
#     print(n_unit, dropout, lr, batch_size, epoch)
    auc = train_and_evaluate_3_layers(n_unit,
                                      dropout,
                                      lr,
                                      batch_size,
                                      epoch,
                                      (pipeline.transform(train_data), train_data[y]),
                                      (pipeline.transform(val_data), val_data[y]),
                                      shape_size)
    print("Number of units: {0}, dropout: {1}, learning rate: {2}, batch size: {3}, epoch: {4}, AUC: {5}".format(n_unit, dropout, lr, batch_size, epoch, auc))
    aucs.append(auc)
max_id = np.argmax(aucs)
print(max_id)
print(np.max(aucs))
print(list(itertools.product(n_units, dropouts, lrs, batch_sizes, epochs))[max_id])


# find optimal param with highest auc
# log = pd.read_csv('log_20190820.txt',sep="\t")
log = pd.read_csv('log_20190822.txt',sep="\t")

log.columns = ['c']
log_auc = log[log['c'].str.contains('AUC:')]

log_auc_param = log_auc.c.str.split(",",expand=True)
log_auc_param.columns = ['n_unit','dropout','lr','batch_size','epoch','auc']
log_auc_param = log_auc_param.apply(lambda x: x.str.split(": ").str[1], axis=0)
log_auc_param['auc'] = log_auc_param['auc'].astype('float')

log_auc_param.loc[log_auc_param['auc'].idxmax()]

# fit NN with the optimal param
inputs = Input(shape=(shape_size,))
X = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(inputs)
X = Dropout(0.2)(X)
X = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(X)
X = Dropout(0.2)(X)
predictions = Dense(1, activation='sigmoid')(X)
model = Model(inputs=inputs, outputs=predictions)
optimizer = optimizers.SGD(lr=0.003, momentum=0.9, decay=1e-5, nesterov=True)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(pipeline.transform(train_data), train_data[y], batch_size=128, epochs=5, validation_split=0.0, shuffle=True, validation_data=(pipeline.transform(val_data), val_data[y]))

#LR - elasticnet
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log', # 'log' means logistic regression
                    penalty='elasticnet', 
                    l1_ratio=0.5, 
                    learning_rate='optimal', 
                    early_stopping=False, 
                    n_iter_no_change=20,
                    class_weight=None)
clf.fit(train_x, train_y)

#LR-L1
clf = SGDClassifier(loss='log', # 'log' means logistic regression
                    penalty='l1', 
                    l1_ratio=1, 
                    learning_rate='optimal', 
                    early_stopping=False, 
                    n_iter_no_change=20,
                    class_weight=None)
clf.fit(train_x, train_y)

#LR-L2
clf = SGDClassifier(loss='log', # 'log' means logistic regression
                    penalty='l2', 
                    l1_ratio=0, 
                    learning_rate='optimal', 
                    early_stopping=False, 
                    n_iter_no_change=20,
                    class_weight=None)
clf.fit(train_x, train_y)

#Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, class_weight=None, verbose=True)
clf.fit(train_x, train_y)

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, random_state=0)
clf.fit(train_x, train_y)

