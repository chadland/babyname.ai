# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:02:29 2019

@author: c202acox
"""
import torch
import pandas as pd
from util import *

def generateSamples(
                    model,
                    all_letters_dict, 
                    all_letters_dict_inverse,
                    all_categories,
                    category_lines,
                    n_categories,
                    n_letters,
                    category='norwegian_female_names',
                    start_character="",
                    temperature=0.2, 
                    use_temperature=True,
                    n_samples=100,
                    store_only_unseen_names=False,
                    csv_file_name='test',
                    store_results=False,
                    print_output=False,
                    model_folder='../model/',
                    data_folder='../model/', 
                    device=torch.device('cpu')):
    

    #storage of names
    new_names = []
    
    #Set device
    model = model.to(device)
    model.device=device
    
    #set model in evaluation mode
    model.eval()
    
    #select category
    category_tensor = categoryTensor(category, all_categories, n_categories)
    
    hidden_init = model.initHiddenPrediction()
    
    
    hidden = hidden_init
    for i in range(n_samples):
        current_letter=''
        output_string =''
        if len(start_character)>0:    
            input_tensor = inputTensor(category_tensor=category_tensor, all_letters_dict=all_letters_dict,line=start_character, n_letters=n_letters, include_start_char=False) 
            current_letter=start_character
            output_string+=start_character
        else:
            input_tensor = inputTensor(category_tensor=category_tensor, all_letters_dict=all_letters_dict,line=start_character, n_letters=n_letters, include_start_char=True)
        input_tensor = torch.stack(input_tensor)
        with torch.no_grad(): 
            #Predictions
            while(current_letter != '<eos>'):
                predictions, hidden = model(input_tensor.to(device), hidden) #.squeeze(1)
            
                if use_temperature:
                    output_dist = predictions.data.view(-1).div(temperature).exp()
                    topi = torch.multinomial(output_dist, 1)[0]
                    letter = all_letters_dict_inverse[topi.item()]
                else:
                    letter_idx, letter = categoryFromOutput(output=predictions,all_letters_dict_inverse=all_letters_dict_inverse)
                    letter = ''.join(letter)
                
                if letter=='<eos>':
                    break
            
                #Add letter 
                output_string+=letter
                
                #Set current letter
                current_letter = letter
                
                #Set input tensor
                input_tensor = inputTensor(category_tensor=category_tensor, all_letters_dict=all_letters_dict,line=letter,n_letters=n_letters,include_start_char=False) 
                input_tensor = torch.stack(input_tensor)
            
            if print_output:
                print(output_string)
            new_names.append(output_string)
                    
    #check if name exist from before
    if store_only_unseen_names:
        result = list(set(new_names)-set(category_lines[category]))
    else:
        result = list(set(new_names))
    
    #Storing samples
    if store_results:
        pd.DataFrame(result).to_csv(data_folder+csv_file_name + ".csv")

    return result