"""
define a function to plot distribution and peak value
"""


#setup environment
import json
import pytz
import datetime
import pandas as pd
import pickle
import os 
import boto3
import s3fs
import pyarrow.parquet as pq
import numpy as np
import warnings
import time
from io import StringIO
import os
from keras.models import load_model
import sys
import yaml
import traceback

import matplotlib
# matplotlib.use('Agg')
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pairwise(group, rpc):
    f, ax = plt.subplots(figsize=(12, 5)) # define figure size
    thekey = rpc_offerkey_dict[rpc]
    if group == 'A':
        lto = score_lto_a.loc[score_lto_a['offer_key']==thekey,'score']
        bau = score_bau_a.loc[score_bau_a['offer_key']==thekey,'score']
    elif group == 'B':
        lto = score_lto_b.loc[score_lto_b['offer_key']==thekey,'score']
        bau = score_bau_b.loc[score_bau_b['offer_key']==thekey,'score']
    ax = sns.distplot(lto, hist=True, kde=False, rug=False, bins=1000, color='red', label='lto') 
    ax = sns.distplot(bau, hist=True, kde=False, rug=False, bins=1000, color='blue', label='bau')
    ax.legend()
    
    #set up scale
    ax.set(xlabel='Probas', ylabel='customers', title="Group "+group+"-"+rpc+" lto bau comparison")
    lto_10q = np.percentile(lto,1)
    bau_10q = np.percentile(bau,1)
    lto_90q = np.percentile(lto,99)
    bau_90q = np.percentile(bau,99)
    lb = min(lto_10q, bau_10q)
    rb = max(lto_90q, bau_10q)
    
    plt.xlim(lb,rb)
    
    #get max y and corresponding x
    height = [h.get_height() for h in ax.patches]
    height_1 = height[:1000]
    height_2 = height[1000:]

    x = [h.get_x() for h in ax.patches]
    x_1 = x[:1000]
    x_2 = x[1000:]

    x1 = x_1[height_1.index(max(height_1))]
    x2 = x_2[height_2.index(max(height_2))]
    h1 = max(height_1)
    h2 = max(height_2)
    
    ax.text(x1, h1, 'x='+str(x1), fontsize=10)
    ax.text(x2, h2, 'x='+str(x2), fontsize=10)
    
    plt.axvline(x1, color = 'black')
    plt.axvline(x2, color = 'black')
    
    f.show()
    
    return x1,x2
    
    
    def plot_allcards(group, offer_type, bins):
    f, ax = plt.subplots(figsize=(12, 5)) # define figure size

    if group == 'A':
        if offer_type == 'bau':
            data = score_bau_a
        elif offer_type == 'lto':
            data = score_lto_a
    elif group == 'B':
        if offer_type == 'bau':
            data = score_bau_b
        elif offer_type == 'lto':
            data = score_lto_b        
    ax = sns.distplot(data.loc[data['offer_key']==rpc_offerkey_dict['0513'],'score'], hist=True, kde=False, rug=False, bins=1000, color='red', label='0513') 
    ax = sns.distplot(data.loc[data['offer_key']==rpc_offerkey_dict['0532'],'score'], hist=True, kde=False, rug=False, bins=1000, color='blue', label='0532') 
    ax = sns.distplot(data.loc[data['offer_key']==rpc_offerkey_dict['USPC'],'score'], hist=True, kde=False, rug=False, bins=1000, color='green', label='USPC') 
    ax = sns.distplot(data.loc[data['offer_key']==rpc_offerkey_dict['USBU'],'score'], hist=True, kde=False, rug=False, bins=1000, color='orange', label='USBU') 

    ax.legend()
    
    #set up scale
    ax.set(xlabel='Probas', ylabel='customers', title="Group "+group+"-"+offer_type+' all cards comparison')
    lb = np.percentile(data['score'],1)
    rb = np.percentile(data['score'],99)
    
    plt.xlim(lb,rb)
    
    #get max y and corresponding x
    height = [h.get_height() for h in ax.patches]
    x = [h.get_x() for h in ax.patches]
    
    for i in range(4):
        height_i = height[i*bins:(i+1)*bins]
        x_i = x[i*bins:(i+1)*bins]
        hi = max(height_i)
        xi = x_i[height_i.index(max(height_i))]
        
        ax.text(xi, hi, 'x='+str(xi), fontsize=10)
        
        plt.axvline(xi, color = 'black')    
    
    f.show()
    
    return ""
