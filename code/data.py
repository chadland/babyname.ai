# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:04:37 2019

@author: c202acox
"""
from __future__ import unicode_literals, print_function, division
from io import open
import pandas as pd
import pickle
import json

#Function to generate data sets
def getBronnoysundData(save=False, 
                       getExisting = True,  
                       category='Programmeringstjenester', 
                       pickle_save_dest='data/Programmeringstjenester.pickle'):
    #Select top 20 company types 
    #df_data_subselect=df_data_subselect[df_data_subselect['naeringskode1.beskrivelse'].isin(df_data_subselect['naeringskode1.beskrivelse'].value_counts()[1:20].index.tolist())]

    
    if getExisting:
        df_data = pickle.load(open( "data/df_data"+'.pickle', "rb" ) )
        df_data_subselect = df_data[df_data['organisasjonsform.kode']=='AS']
        df_data_subselect = df_data_subselect[df_data_subselect['naeringskode1.beskrivelse']==category]
        df_data_subselect = df_data_subselect[['navn','naeringskode1.beskrivelse']]
        df_data_subselect = df_data_subselect.rename(index=str, columns={'navn':'name','naeringskode1.beskrivelse': 'category'})
        pickle.dump(df_data_subselect, open( pickle_save_dest, "wb" ) )
        return df_data_subselect
    else:
        with open('data/enheter_alle.json',encoding="utf8") as f:
            data = json.load(f)
        df_data = pd.io.json.json_normalize(data)
        df_data.head()
        df_data.columns
        if save==True:
            
            pickle.dump(df_data, open( "data/df_data"+'.pickle', "wb" ) )
        return df_data

def getBlackSpeech( pickle_save_dest='data/blackspeech.pickle'):
    data = pd.read_csv('data/black-speech.txt', header=None)
    data.columns = ["name"]
    data['category'] = 'mordor_speech'
    pickle.dump(data, open( pickle_save_dest,"wb" ) )
    return data

def getIslandicNames( pickle_save_dest='data/icelandic_male_names.pickle'):
    data = pd.read_csv('data/ICELANDIC_MALE_NAMES.csv', header=None)
    data.columns = ["name"]
    data['category'] = 'icelandic_names'
    pickle.dump(data, open( pickle_save_dest,"wb" ) )
    return data
    
def getRapNames( pickle_save_dest='data/rap_names.pickle'):
    data = pd.read_csv('data/rap_names.txt', header=None)
    data.columns = ["name"]
    data['category'] = 'rap_names'
    pickle.dump(data, open( pickle_save_dest,"wb" ) )
    return data

def getNorwegianFemaleNames( pickle_save_dest='../data/norwegian_female_names.pickle'):
    data = pd.read_csv('../data/NORWEGIAN_FEMALE_NAMES_997.txt', encoding = "utf-16",
                                        sep='\t',header=None)
    data.columns = ["name"]
    data['category'] = 'norwegian_female_names'
    pickle.dump(data, open( pickle_save_dest,"wb" ) )
    return data

def getNorwegianMaleNames( pickle_save_dest='../data/norwegian_male_names.pickle'):
    data = pd.read_csv('../data/NORWEGIAN_MALE_NAMES_861.txt', encoding = "utf-16",
                                        sep='\t',header=None)
    data.columns = ["name"]
    data['category'] = 'norwegian_male_names'
    pickle.dump(data, open( pickle_save_dest,"wb" ) )
    return data

def getMixNorwegianAndIcelandicMaleNames( pickle_save_dest='../data/norwegian_icelandic_male_names.pickle'):
    data_nor = pd.read_csv('../data/NORWEGIAN_MALE_NAMES_861.txt', encoding = "utf-16",
                                        sep='\t',header=None)
    data_nor.columns = ["name"]
    data_nor['category'] = 'norwegian_male_names'
    data_icl = pd.read_csv('../data/ICELANDIC_MALE_NAMES.csv', header=None)
    data_icl.columns = ["name"]
    data_icl['category'] = 'icelandic_male_names'
    data=pd.concat([data_nor, data_icl])
    pickle.dump(data, open( pickle_save_dest,"wb" ) )
    return data
