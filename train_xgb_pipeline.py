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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
import xgboost as xgb
import io

warnings.filterwarnings("ignore")

%load_ext autoreload
%autoreload 2

pd.options.display.max_columns = None


s3 = boto3.resource('s3')
s3c = boto3.client('s3')
s3sys = s3fs.S3FileSystem()


train_name = ''
test_name = ''
val_name = ''

#read parquet data from S3
train_data = pq.ParquetDataset(train_name, filesystem=s3sys).read_pandas().to_pandas()
val_data = pq.ParquetDataset(val_name, filesystem=s3sys).read_pandas().to_pandas()
test_data = pq.ParquetDataset(test_name, filesystem=s3sys).read_pandas().to_pandas()


def get_xgb_datasets(train_data, val_data, test_data):
    xg_train = train_data.copy()
    xg_val = val_data.copy()
    xg_test = test_data.copy()
    
    pipeline, x, y = __fit_pipeline(xg_train)

    xg_train_x = pipeline.transform(xg_train)
    xg_train_y = xg_train[y]

    xg_val_x = pipeline.transform(xg_val)
    xg_val_y = xg_val[y]

    xg_test_x = pipeline.transform(xg_test)
    xg_test_y = xg_test[y]
    
    return xg_train_x, xg_train_y, xg_val_x, xg_val_y, xg_test_x, xg_test_y, pipeline, x, y
    
def get_xgb_datasets(val_data, test_data, pipeline):
    xg_val = val_data.copy()
    xg_test = test_data.copy()

    xg_val_x = pipeline.transform(xg_val)
    xg_val_y = xg_val[y]

    xg_test_x = pipeline.transform(xg_test)
    xg_test_y = xg_test[y]
    
    return xg_val_x, xg_val_y, xg_test_x, xg_test_y
    
    
def __fit_pipeline(data):

    #define x variables and Y variable
    y = 'response_var'
    x = []
    
    categorical_features = []

    numeric_features = x
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=True, with_std=True))])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='error'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    pipeline.fit(data)
    
    return pipeline, x, y
    
#load a pipeline
pipeline = pickle.loads(s3.Bucket(BUCKET).Object('some_pipeline_path').get()['Body'].read())

#transform data
xg_train_x = pipeline.transform(xg_train_sample)
xg_train_y = xg_train_sample[y]

xg_val_x, xg_val_y, xg_test_x, xg_test_y= get_xgb_datasets(val_data, test_data, pipeline)

#train xgb
eval_set = [(xg_train_x,xg_train_y),(xg_val_x,xg_val_y)]
xg_class_capped_unweighted = xgb.XGBClassifier(learning_rate =0.01,
                                 n_estimators=400,
                                 max_depth=23,
                                 min_child_weight=1,
                                 gamma=0,
                                 subsample=0.8,
                                 reg_lambda=2,
                                 colsample_bytree=0.4,
                                 objective= 'binary:logistic',
                                 nthread=4,
                                 scale_pos_weight=1,
                                 seed=207)
                                 
xg_class_capped_unweighted.fit(X=xg_train_x, y=xg_train_y, verbose=True, eval_set=eval_set, eval_metric=['auc', 'error','logloss'])
 

#plot learning curve
results = xg_class_capped_unweighted.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

sns.set_style('darkgrid')

# plot auc
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Validation')
ax.legend()
plt.ylabel('AUC')
plt.title('XGBoost AUC, outlier capped')
plt.show()

# plot log loss
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss, outlier capped')
plt.show()

# plot classification error
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Validation')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error, outlier capped')
plt.show()


#get feature importance with one-hot encoded features
enc = OneHotEncoder(handle_unknown='error')
enc.fit(val_data_capped[['channel', 'rpc_cd', 'CUSTOMER_STATUS']])
x_copy = x.copy()
for cat in enc.categories_:
    for i in cat:
        x_copy.append(i)
        
# feature importance
importance = pd.Series(xg_class_capped_unweighted.feature_importances_, index=x_copy).sort_values(ascending=False)
# print(importance.to_string())
f, ax = plt.subplots(figsize=(8, 40)) # define figure size
ax = sns.barplot(x=importance.values,y=importance.index)
ax.set(xlabel='Weight', ylabel='', title='Feature Importance from XGBoost')
f.show()
f.tight_layout()
img_data = io.BytesIO()
f.savefig(img_data, format='png', dpi = 100)
img_data.seek(0)
bucket_s3.put_object(Body=img_data, ContentType='image/png', Key='some_path.png')

# save xg_class to model locally
pickle.dump(xg_class_capped_unweighted, open("model_name.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
# save pipeline locally
pickle.dump(pipeline, open("pipeline_name.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# save model and pipeline to S3
with s3sys.open("s3_model_path.pkl",'wb') as f:
    pickle.dump(xg_class_capped_unweighted, f, protocol=pickle.HIGHEST_PROTOCOL)

with s3sys.open("s3_pipeline_path.pkl",'wb') as f:
    pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    
#dump summary json
summary_json['a'] = ''
summary_json['b'] = ''
summary_json['c'] = ''

s3_object = s3.Object(bucket, "s3_path_model_summary.json")
s3_object.put(Body=json.dumps(summary_json))
