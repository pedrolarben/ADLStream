#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm


# In[2]:


_RESULT_DIR = ['../DSClassificationResults/DSClassificationResults_keras', '../DSClassificationResults/DSClassificationResults_sklearn', '../DSClassificationResults/DSClassificationResults_MOA'] 
_MODEL = 'keras_parallel_3_Dilated_Conv'
_METRICS_FILE_NAME = 'metrics.csv'
_DATA_FILE_NAME = 'data.csv'
_RES_FILE_PATH = './notebooks/files/benchmark_sensitivityAnalysis.csv'
#_DATASETS = sorted(os.listdir(_RESULT_DIR))



# In[3]:


def prequential_accuracy(data, fading_factor=0.98, chunk_size=10):
    # get accuracy for each instance
    accuracy = (data[data.columns[data.shape[1]-2]]==data[data.columns[data.shape[1]-1]]).astype("int")
    # every chunks should have the same size
    accuracy = accuracy.iloc[:accuracy.shape[0]-(accuracy.shape[0]%chunk_size)]
    # compute accuracy for each chunk
    accuracy = [accuracy.iloc[start:end].mean() 
                for (start,end) in [
                    (i*chunk_size, (i+1)*chunk_size) for i in range(int(accuracy.shape[0]/chunk_size))
                ]
               ]
    # compute fading factor for each chunk
    i = len(accuracy)
    fading_factor_chunks = [fading_factor**(i-k) for k in range(1, i+1)]
    fading_factor_sum = sum(fading_factor_chunks)
    # Compute the final accuracy: sum the faded accuracies 
    accuracy_faded = sum([(fading_factor_chunks[k]*accuracy[k])/fading_factor_sum for k in range(len(accuracy))])
    return accuracy_faded


def prequential_kappa(data, fading_factor=0.98, chunk_size=10):
    # every chunks should have the same size
    data_clean =  data [[data.columns[data.shape[1]-2], data.columns[data.shape[1]-1]]].astype(int)
    data_clean.columns = ["Class", "Prediction"]
    # get kappa for each chunks
    kappa = [cohen_kappa_score(data_clean.iloc[start:end]["Class"], data_clean.iloc[start:end]["Prediction"]) 
              for (start,end) in [
                  (i*chunk_size, (i+1)*chunk_size) for i in range(int(data_clean.shape[0]/chunk_size))
              ]
             ]
    # Change NaN for 1 (kappa is nan when there is only one class in the chunk)
    kappa = [1. if math.isnan(k) else k for k in kappa]
    # compute fading factor for each chunk
    i = len(kappa)
    fading_factor_chunks = [fading_factor**(i-k) for k in range(1, i+1)]
    fading_factor_sum = sum(fading_factor_chunks)
    # compute final kappa: sum the faded kappa
    kappa_faded = sum([(fading_factor_chunks[k]*kappa[k])/fading_factor_sum for k in range(len(kappa))])
    return kappa_faded


# In[4]:


errors = []
metrics_ls = []

for result_dir in _RESULT_DIR:
    datasets = sorted(os.listdir(result_dir))
    for dataset in tqdm(datasets, desc=result_dir):
        models = sorted(os.listdir(os.path.join(result_dir, dataset)))
        for model in models:
            # read files 
            metrics_file_path = os.path.join(result_dir, dataset, model, _METRICS_FILE_NAME)
            data_file_path = os.path.join(result_dir, dataset, model, _DATA_FILE_NAME)
            try:
                if 'MOA' in model:
                    data = pd.read_csv(data_file_path,header=None)
                else:
                    data = pd.read_csv(data_file_path,header=0)
                metrics = pd.read_csv(metrics_file_path)
                metrics.columns = [c.lower() for c in metrics.columns]
            except Exception as e:
                errors.append((model,dataset, str(e)))
                continue
            
            # compute metrics
            num_instances = data.shape[0]
            num_attributes = data.shape[1]-2
            num_classes = len(pd.unique(data[data.columns[data.shape[1]-2]].astype(int)))
            
            accuracy = prequential_accuracy(data)
            kappa = prequential_kappa(data)
            
            train_time_mean = metrics['train_time_s'].mean() if 'train_time_s' in metrics else metrics['train_time'].mean()
            test_time_mean = metrics['test_time_s'].mean() if 'test_time_s' in metrics else metrics['test_time'].mean()
            total_time_mean = (metrics['train_time_s'] + metrics['test_time_s']).mean() if 'test_time_s' in metrics else (metrics['train_time'] + metrics['test_time']).mean()
            
            train_time = metrics['train_time_s'].sum() if 'train_time_s' in metrics else metrics['train_time'].sum()
            test_time = metrics['test_time_s'].sum() if 'test_time_s' in metrics else metrics['test_time'].sum()
            total_time = (metrics['train_time_s'].sum() + metrics['test_time_s'].sum() if 'parallel' not in model else max(metrics['train_time_s'].sum(), metrics['test_time_s'].sum())) if 'test_time_s' in metrics else (metrics['train_time'].sum() + metrics['test_time'].sum() if 'parallel' not in model else max(metrics['train_time'].sum(), metrics['test_time'].sum()))
                
            metrics_summary = {'dataset':dataset,
                      'classifier':model,
                      'instances': num_instances,
                      'attributes': num_attributes,
                      'classes': num_classes,
                      'accuracy': accuracy,
                      'kappa': kappa,
                      'train_time_mean': train_time_mean,
                      'test_time_mean': test_time_mean,
                      'total_time_mean': total_time_mean,
                      'train_time': train_time,
                      'test_time': test_time,
                      'total_time': total_time        
                     }
            metrics_ls.append(metrics_summary)
        

res = pd.DataFrame(metrics_ls, columns=['dataset','classifier','instances', 'attributes', 'classes','accuracy', 'kappa', 'train_time_mean', 'test_time_mean', 'total_time_mean', 'train_time', 'test_time', 'total_time'])
res = res.sort_values(by=['dataset', 'classifier'])
res.to_csv(_RES_FILE_PATH, index=False)

if errors:
    print("Errors: " , errors)
    


# In[5]:


errors


# In[6]:


## DEPRECATED - DO NOT USE
#_BATCH_SIZE = 10
#_NUM_BATCH_FED = 40
#
#errors = []
#metrics_ls = []
#
#for result_dir in _RESULT_DIR:
#    datasets = sorted(os.listdir(result_dir))
#    for dataset in datasets:
#        models = sorted(os.listdir(os.path.join(result_dir, dataset)))
#        for model in models:               
#            file_path = os.path.join(result_dir, dataset, model, _FILE_NAME)
#            
#            try:
#                metrics = pd.read_csv(file_path)
#            except Exception:
#                errors.append(model + " - " + dataset)
#                continue
#            
#            if metrics['total'].max() <= _BATCH_SIZE*_NUM_BATCH_FED:
#                continue
#            
#            metrics = metrics[metrics['total'] > _BATCH_SIZE*_NUM_BATCH_FED]
#            
#            if model.startswith("MOA"):
#                metrics['accuracy'] = metrics['accuracy']/100. 
#                
#            metrics_summary = {'dataset':dataset,
#                      'classifier':model,
#                      'total':metrics['total'].max(),
#                      'tp':metrics['tp'].mean() if 'tp' in metrics.columns else "",
#                      'tn':metrics['tn'].mean() if 'tn' in metrics.columns else "", 
#                      'fp':metrics['fp'].mean() if 'fp' in metrics.columns else "",
#                      'fn':metrics['fn'].mean() if 'fn' in metrics.columns else "",
#                      'precision':metrics['precision'].mean() if 'precision' in metrics.columns else "",
#                      'recall':metrics['recall'].mean() if 'recall' in metrics.columns else "",
#                      'f1':metrics['f1'].mean() if 'f1' in metrics.columns else "",
#                      'fbeta':metrics['fbeta'].mean() if 'fbeta' in metrics.columns else "",
#                      'accuracy':metrics['accuracy'].mean() if 'accuracy' in metrics.columns else "",
#                      'train_time_s':metrics['train_time'].sum() if 'train_time' in metrics.columns else "",
#                      'test_time_s':metrics['test_time'].sum() if 'test_time' in metrics.columns else ""
#                     }
#            metrics_ls.append(metrics_summary)
#
#res = pd.DataFrame(metrics_ls, columns=['dataset','classifier','total','tp','tn','fp','fn','precision','recall','f1','fbeta','accuracy','train_time_s','test_time_s'])
#res = res.sort_values(by=['dataset', 'classifier'])
#res.to_csv(_RES_FILE_PATH, index=False)
#
#if errors:
#    print("Errors: " + str(errors))
#    

