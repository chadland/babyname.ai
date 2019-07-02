# -*- coding: utf-8 -*-
"""
Generating names with 

"""
%reset
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import pandas as pd
import pickle
import json
from pprint import pprint
from random import randint
import random
import argparse
import numpy as np
import os
import sys
from generate import *
from util import *
from model_lib import *
from data import *
from importlib import reload 
import time

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--input_df', type=str, default='../data/norwegian_male_names.pickle')
argparser.add_argument('--n_epochs', type=int, default=500)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--hidden_size', type=int, default=128)
argparser.add_argument('--dropout_fact', type=int, default=0.5)
argparser.add_argument('--learning_rate', type=int, default=0.0005)
argparser.add_argument('--store_model', type=bool, default=True)
argparser.add_argument('--model_prefix', type=str, default='NORWEGIAN_MALE_NAMES_')
argparser.add_argument('--model_folder', type=str, default='../model/')
argparser.add_argument('--data_folder', type=str, default='../data/')
argparser.add_argument('--model_type', type=str, default='LSTM')
argparser.add_argument('--model_descripton', type=str, default='Norwegian female name generator')

#argparser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
args, _ = argparser.parse_known_args()

#preprocess data
all_letters, all_letters_dict,all_letters_dict_inverse, n_letters, category_lines ,all_categories, all_categories_dict , n_categories  = preprocessData(args.input_df)        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
#Create model
model = LSTM(input_size=n_letters+n_categories, 
           hidden_size=args.hidden_size, 
           output_size=n_letters,
           n_layers=args.n_layers,
           dropout_fact=args.dropout_fact, 
           batch_size=args.batch_size,
           device=device )

#Set devices to model     
model = model.to(device)     
criterion = nn.NLLLoss() 
optimizer = torch.optim.Adam(model.parameters())
criterion.to(device)

#Create batch data set
iterator = create_batch_data(all_categories,
                      n_categories,
                      all_letters_dict,
                      n_letters,
                      args.input_df, 
                      batch_size=args.batch_size)

#Train model
for i in range(args.n_epochs):
    #Crate new batches for each iteration as we want to shuffle the names each time
#    iterator = create_batch_data(all_categories,
#                      n_categories,
#                      all_letters_dict,
#                      n_letters,
#                      args.input_df, 
#                      batch_size=args.batch_size)
    avg_loss, avg_accuracy =  train(model, iterator, optimizer, criterion, all_letters_dict_inverse, print_each_batch_result=False )
    print("Epoch: "+ str(i))
    print("Avg Epoch Loss: " + str(avg_loss))
    print("Avg Accuracy: " + str(avg_accuracy))


#Model name post fix
model_name_postfix= str(args.model_type
    +"_NLAYERS_" + str(args.n_layers)
    +"_HIDDENSIZE_"+str(args.hidden_size)
    +"_N_EPOCS_"+str(args.n_epochs)
    +"_DEVICE_"+ device.type.upper())

#Generate samples
#For high temperatures (τ→∞), all [samples] have nearly the same probability and the lower the temperature, the more expected rewards affect the probability. 
#For a low temperature (τ→0+), the probability of the [sample] with the highest expected reward tends to 1.
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
                n_samples=500,
                store_only_unseen_names=False,
                #csv_file_name=args.model_prefix+model_name_postfix+"_"+all_categories[0].upper(),
                store_results=False)
                #=args.model_folder,
                #data_folder=args.data_folder)    

#Test similarity against word
df_top20_words = word_similarity("Ruben", generated_samples)
print("Levenshtein Measure Top 20 Similar words to June")
print(df_top20_words.word_to_compare_against)

df_top20_words = word_similarity("Ruben", generated_samples, sim_measure='SequenceMatcher')
print("SequenceMatcher Measure Top 20 Similar words to Sophie")
print(df_top20_words.word_to_compare_against)

df_top20_words = word_similarity("Christer", generated_samples, sim_measure='Jaro-Winkler')
print("Jaro-Winkler Measure Top 20 Similar words to June")
print(df_top20_words.word_to_compare_against)

df_top20_words = word_similarity("Christine", generated_samples, sim_measure='Hamming')
print("Hamming Measure Top 20 Similar words to June")
print(df_top20_words.word_to_compare_against)

#Get all names for storing
names = []
for k in category_lines:
    names = names + category_lines[k]
    
#Save Model
store_model(model=model,
           optimizer=optimizer,
           model_folder=args.model_folder,
           model_name=args.model_prefix+model_name_postfix,
           model_desc=args.model_descripton,
           all_letters=all_letters, 
           all_letters_dict=all_letters_dict,
           all_letters_dict_inverse=all_letters_dict_inverse, 
           n_letters=n_letters,
           category_lines=category_lines ,
           all_categories=all_categories, 
           all_categories_dict=all_categories_dict , 
           n_categories=n_categories,
           train_loader=iterator,
           input_df=args.input_df,
           n_epochs=args.n_epochs,
           batch_size=args.batch_size,
           n_layers=args.n_layers,
           hidden_size=args.hidden_size,
           dropout_fact=args.dropout_fact,
           learning_rate=args.learning_rate,
           model_type=args.model_type,
           avg_loss=avg_loss,
           avg_accuracy=avg_accuracy,
           training_names=names)

#Training Names
model, all_letters_dict, all_letters_dict_inverse, all_categories,category_lines,n_categories,n_letters, training_names=load_model(file_name='../model/' + 'NORWEGIAN_MALE_NAMES_LSTM_NLAYERS_2_HIDDENSIZE_128_N_EPOCS_500_DEVICE_CUDA.pt')

#Generate samples
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
                n_samples=500,
                store_only_unseen_names=False,
                #csv_file_name=args.model_prefix+model_name_postfix+"_"+all_categories[0].upper(),
                store_results=False)
                #=args.model_folder,
                #data_folder=args.data_folder)    

df_top20_words = word_similarity("Liam", generated_samples)
print("Levenshtein Measure Top 20 Similar words to June")
print(df_top20_words.word_to_compare_against)


#TODO
#Sounds like a proper name network
#Sounds like a startup name network
#Sounds like measure - SOUNDEX
#Max number of letters
#Filter out names with three concecutive letters

#Create seperate genearator
    #Load model
    #Generate
#Create international score (how well does the name sound in English / Chineese)
#Deploy in sagemaker
#Implement in sagemaker
#Star Wars names
#How unique is my name    
