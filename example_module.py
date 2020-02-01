"""
An example python module that can be used in orchestration.py
"""

#setup environment
import json
import pytz
import datetime
import pandas as pd
import os 
import boto3
import numpy as np
import warnings
import time
import yaml
import s3fs
import pyarrow.parquet as pq
import sys
import traceback
from skmultilearn.model_selection import iterative_train_test_split


path_array = os.getcwd().split("/")
separator = "/"
sys.path.append(separator.join(path_array[0:len(path_array)-1])+'/config')
import constants

warnings.filterwarnings("ignore")

#set eastern time
eastern = pytz.timezone('US/Eastern')
d = datetime.datetime.now(eastern)

s3fs = s3fs.S3FileSystem()
s3 = boto3.resource('s3')
s3c = boto3.client('s3')
bucket_s3 = s3.Bucket(constants.BUCKET)

#read in config file
if path_array[len(path_array)-1] != constants.REPO:
    with open(separator.join(path_array[0:len(path_array)-1])+'/'+constants.CONFIG, 'r') as ymlfile:
        configfile = yaml.safe_load(ymlfile)
else:
    with open(separator.join(path_array[0:len(path_array)])+'/'+constants.CONFIG, 'r') as ymlfile:
        configfile = yaml.safe_load(ymlfile)


def fun1(input1,input2,input3):
    status = "SUCCESS"
    message = None
    body = None
    total_time = str(0)
    
    try:
        #start timer
        start = time.time()
        
        #verify data is in the right format
        try:
            datetime.datetime.strptime(some_input_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Incorrect date format, should be 'YYYY-MM-DD'")
            
        # function body
        #
        #
        #
        
        end = time.time()
        total_time = end-start
        
        #save number of records and time to log
        message = some_string
        body = str(total_time) + "consumed to finish this run"
        
    except Exception as e:
        status = "FAILURE"
        message = traceback.format_exc()
        # message = str(e)
    finally:
        resp = {}
        resp['status'] = status
        resp['message'] = message
        resp['body'] = body
        print(resp, flush=True)
        print()
        return json.dumps(resp)
def __subfun(input1,input2):
    # some code
    # 
    return something

if __name__== "__main__":
    #read runtime parameters
    param1 = ''
    param2 = ''
    param3 = ''
    
    if len(sys.argv) >= 7:
        param1=sys.argv[1]
        param2=sys.argv[3]
        param3=sys.argv[7]
        
    #call function  
    resp = json.loads(assign_groups(param1, param2, param3))
    if resp['status'] == "FAILURE":
        raise RuntimeError(resp['message'])
