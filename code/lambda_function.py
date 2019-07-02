import boto3
import os
import pickle
import json
import numpy as np
import torch
import pandas as pd
from util import *
from torch.autograd import Variable
from generate import *


ACCESS_KEY = os.environ.get('ACCESS_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
MODEL_KEY = os.environ.get('MODEL_KEY') # model.pth
PARAMS_KEY = os.environ.get('PARAMS_KEY') # params.pkl
model_tmp = os.environ.get('model_tmp') # 20
#model_tmp = '../model/'

s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

# Download model from S3 if model is not already present
def downloadFiles(model_name='models/NORWEGIAN_FEMALE_NAMES_LSTM_NLAYERS_2_HIDDENSIZE_128_N_EPOCS_500_DEVICE_CUDA.pt'):
    if not os.path.isfile(model_tmp+model_name):
        if not os.path.exists(model_tmp+model_name.split('/')[0]):
            os.makedirs(model_tmp+model_name.split('/')[0])
        s3_client.download_file(BUCKET_NAME, model_name, model_tmp+model_name)
    
# Get multiple samples from one category and multiple starting letters
def getPredictions(input_txt="Christer", name_creativity=0.8, model_name='models/NORWEGIAN_FEMALE_NAMES_LSTM_NLAYERS_2_HIDDENSIZE_128_N_EPOCS_500_DEVICE_CUDA.pt'):
    model, all_letters_dict, all_letters_dict_inverse, all_categories,category_lines,n_categories,n_letters,training_names  = load_model(model_tmp+model_name)
    device = torch.device('cpu')
    model = model.to(device)
    generated_samples=generateSamples(
                model,
                all_letters_dict, 
                all_letters_dict_inverse,
                all_categories,
                category_lines,
                n_categories,
                n_letters,
                category=all_categories[0],
                start_character="",
                temperature=0.8, 
                use_temperature=True,
                n_samples=750,
                store_only_unseen_names=False,
                #csv_file_name=args.model_prefix+model_name_postfix+"_"+all_categories[0].upper(),
                store_results=False)
                #=args.model_folder,
                #data_folder=args.data_folder)    
    
    #AI names         
    df_AI_names = word_similarity(input_txt, generated_samples, return_top_n=int(100*name_creativity))        
    
    #Regular names
    df_regular_names= word_similarity(input_txt, training_names, return_top_n=int(100*(1-name_creativity)))
    
    #Merge two data frame
    names = pd.concat([df_AI_names,df_regular_names]).sort_values(by=['similarity'], ascending=False).word_to_compare_against.tolist()
    
    return names

def lambda_handler(event, context):

    #Print parameters
    print('event:', event)
    
    #Download model file
    downloadFiles(event['model_name'])
    
    #Preprocess Name
    event['first_name']=event['first_name'].lower().title()
    
    #Get predictions and extract values
    output_names = getPredictions(input_txt=event['first_name'],
                                  name_creativity=float(event['name_creativity']),
                                  model_name=event['model_name'])
    #output_names = getPredictions()
    
    print('output_names', output_names)
    
    # return results formatted for AWS API Gateway
#    result = {"statusCode": 200, \
#            "headers": {"Content-Type": "application/json"}, \
#             "body": json.dumps(output_names)}
    
    result = {"output_names": output_names
            }
    
    print('result',result)
    
    return result



