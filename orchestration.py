"""
Python wrapper to kickoff the other modules. It parses command line arguments and execute the python modules in a preferred sequence
runtime argument takes json like input and the value can be assigned to variables. for example, a run.sh to run this orchestration script could be:
nohup pythonrOrchestration.py '{"mode": ["ETL","preprocess","train","score","create","QA"], "obs_date": "2020-01-01"}' &> log/log_20200101.txt &
"""

#setup environment
import os 
import time
import sys
import pandas as pd
import datetime 
import json
import boto3
import s3fs
from io import StringIO
import sys
import traceback
import pytz
import yaml

# import python modules, for example
import constants
import ETL
import preprocess
import train
import scorefactory
import optimization
import creation
import QA

s3c = boto3.client('s3')
#read in config file
with open(os.getcwd()+"/"+constants.CONFIG, 'r') as ymlfile:
    configfile = yaml.safe_load(ymlfile)

if __name__== "__main__":

    #set time
    eastern = pytz.timezone('US/Eastern')
    d = datetime.datetime.now(eastern)
    
    # read in runtim arguments
    runtime_arguments = json.loads(sys.argv[1])
   
    if len(runtime_arguments) != 3:
        raise ValueError('Incorrect number of arguments. Please try again using this format: python Orchestration.py {"mode": [mode], "obs_date": [YYYY-MM-DD] "customer_keys_location" : [path]}')
    
    mode = runtime_arguments['mode']

    # Unique run ID.
    run_id = d.strftime("%Y%m%d%H%M%S")
    
    # Init
    obs_date = runtime_arguments['obs_date']
    status = "SUCCESS"
    message = None
    bucket = constants.BUCKET
    delivery_date = (pd.to_datetime(obs_date) + pd.to_timedelta(7, unit = 'days')).strftime("%Y%m%d")
    prefix = configfile['locations']['production_dir']+'/delivery_'+delivery_date+"/"+str(run_id)
    total_time = str(0)
    body = None
    RUN_ENV = 'DEV'
    
    #set s3 object for logging
    s3 = boto3.resource('s3')
    s3_object = s3.Object(bucket, prefix + "/run_summary.json")
    s3c = boto3.client('s3')
    log = s3.Object(bucket, prefix + '/log.txt')
    buffer = StringIO()
    
    try:
        start = time.time()
        if 'training' in mode:
            print('Executing training', flush=True)

            #copy config to s3 folder
            s3.meta.client.upload_file(os.getcwd()+"/"+constants.CONFIG, bucket, some_s3_path+'/config.yml')
            
            # copy files from an s3 path to another
            s3.meta.client.copy_object(CopySource = {'Bucket': from_this_s3_bucket,'Key': from_this_s3_path, Bucket = to_this_s3_bucket, Key= to_this_s3_path)

            # execute modules
            resp = json.loads(some_python_module.some_function(some_input_param))
            if resp['status'] == "FAILURE":
                raise RuntimeError(resp['message'])
           
        if some_module in some_runtime_arg:
            print('Executing some module', flush=True)
            log_text = ""
            # execute module
            resp = json.loads(some_python_module.some_function(some_input_param))
            if resp['status'] == "FAILURE":
                raise RuntimeError(resp['message'])
            else:
                json.dump(resp, buffer, indent=4, sort_keys=True)
                log_text = log_text + "some module ran successfully" + "\n"
                log.put(Body = log_text)
        
        #calculate script runtime
        end = time.time()
        total_time = end-start
        body = "orchestration for run id "+ str(run_id) +" ran in " + str(total_time) + " seconds" 
        
    except Exception as e:
        status = "FAILURE"
        message = traceback.format_exc()
        # message = str(e)
    finally:
        resp = {}
        resp['status'] = status
        resp['message'] = message
        resp['body'] =  body 
        
        # Send json response into S3 run bucket as run summary
        json.dump(resp, buffer, indent=4, sort_keys=True)
        s3_object.put(Body=buffer.getvalue())
        print(resp, flush=True)
