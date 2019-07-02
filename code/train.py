# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:03:52 2019

@author: c202acox
"""
from __future__ import unicode_literals, print_function, division
from io import open
import pickle

#Preprocess data
def preprocessData(input_df):  
    input_df = pickle.load(open( input_df, "rb" ) )
    
    #Find unique letters
    letters=set()
    for idx, row in input_df.name.iteritems():
        #print(r)
        letters = letters | set(row)
    letters    
    
    all_letters = ''.join(list(letters))
        #all_letters = string.ascii_letters + "ÆØÅæøå.,;'-"
    len(all_letters)
    n_letters = len(all_letters) + 3 # Plus <PAD> <SOS>, <EOS> and  marker
    
    #Create letters dict
    all_letters_dict ={}
    all_letters_dict['<pad>'] = 0
    all_letters_dict['<sos>'] = 1
    all_letters_dict['<eos>'] = 2
    n = len(all_letters_dict)-1
    
    for s in all_letters:
        n=n+1
        all_letters_dict[s] = n
    
    #Inverse dictionary
    all_letters_dict_inverse = {v: k for k, v in all_letters_dict.items()}
    
    
    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    #def unicodeToAscii(s):
    #    return ''.join(
    #        c for c in unicodedata.normalize('NFD', s)
    #        if unicodedata.category(c) != 'Mn'
    #        and c in all_letters
    #    )
    
    # Build the category_lines dictionary, a list of lines per category
    category_lines = {}
    all_categories = []
    for value in input_df['category'].unique():
        category = value
        all_categories.append(category)
        category_lines[category] = input_df[input_df['category']==value].name.tolist()
        
    n_categories = len(all_categories)
    
    #Category dic
    all_categories_dict = {k: v for v, k in enumerate(all_categories)}
    
    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
            'from https://download.pytorch.org/tutorial/data.zip and extract it to '
            'the current directory.')
    
    print('# categories:', n_categories, all_categories)

    return all_letters, all_letters_dict,all_letters_dict_inverse, n_letters, category_lines ,all_categories, all_categories_dict , n_categories 
