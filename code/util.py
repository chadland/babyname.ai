# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:02:20 2019

@author: c202acox
"""

from __future__ import unicode_literals, print_function, division
from io import open
import pickle
import torch
import torch.utils.data as data_utils
import numpy as np
from difflib import SequenceMatcher
import pandas as pd
##import Levenshtein
import jellyfish
from model_lib import *

train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    


#Preprocess data
def preprocessData(input_df): 
    
    """Process data

    Returns objects necessary for training the model
    
    Args:
        input_df (str) - path to pickled pandas with the following columns; name (str) and category (str)

    Returns:
        all_letters (str) - all letters available
        all_letters_dict (dict) - all letters dict,
        all_letters_dict_inverse (dict) - all letters reversed dict, 
        n_letters (int) - number of letters, 
        category_lines - category lines,
        all_categories - all categories, 
        all_categories_dict - all categories dict, 
        n_categories - number of categories 

    """
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

# One-hot vector for category
def categoryTensor(category, all_categories, n_categories):
    """Category tensor

    Returns category tensor
    
    Args:
        category (str) - category, 
        all_categories (list) - list of all categories, 
        n_categories (int) - number of categories

    Returns:
        a one hot encoded tensor containing the index of the category
    """
    li = all_categories.index(category)
    tensor = torch.zeros(n_categories, dtype=torch.long)
    tensor[li] = 1
    
    return tensor

def inputTensor(category_tensor,all_letters_dict,n_letters ,line="Wu Tang Clan", include_start_char=True):
    
    """Input tensor

    Returns a list of input tensors for a word
    
    Args:
        category_tensor (tensor) - category tensor, 
        all_letters_dict (dict) - all letters dict
        n_letters (int) - number of letters
        line (str) - line string
        include_start_charcter (bool)  - whether to include a <sos> tensor
        

    Returns:
        a list of tensors containing category and letters in a name 
    """
    
    input_tensor_list = []
    
    if include_start_char:
        #First line should be start of name
        input_tensor = torch.zeros(n_letters, dtype = torch.long)
        input_tensor[all_letters_dict['<sos>']] = 1
    
        input_tensor= torch.cat([category_tensor, input_tensor])
        input_tensor.size()
            
        input_tensor_list.append(input_tensor)
    
    #Iterate through letters
    for letter in line:
        letter_tensor = torch.zeros( n_letters, dtype = torch.long)
        letter_tensor[all_letters_dict[letter]]=1
        input_tensor_list.append(torch.cat([category_tensor, letter_tensor]))
        
    
    return input_tensor_list


# LongTensor of second letter to end (EOS) for target
def targetTensor(all_letters_dict, line="Wu-Tang Clan"):
    """Target tensor

    Returns target tensor with indecies of letters in name shifted by one to the right 
    
    Args:
        all_letters_dict (dict) - all letters dict
        line (str) - line string
        

    Returns:
        a list of tensor containing indecies of targets
    """
    
    letter_indexes = [torch.LongTensor([all_letters_dict[line[li]]]) for li in range(len(line))]   
    letter_indexes.append(torch.LongTensor([all_letters_dict['<eos>']])) # EOS

    #torch.LongTensor(letter_indexes).size()

    return letter_indexes


def create_batch_data(all_categories,
                      n_categories,
                      all_letters_dict,
                      n_letters,
                      input_df, 
                      batch_size=64, 
                      shuffle_pandas=True):
    """Batch data

    Returns a data loader object of batch data
    
    Args:
        all_categories (list) - all categories,
        n_categories (int) - number of categories,
        all_letters_dict (dict) - all letters dict,
        input_df (str) - path string of pickled pandas data frame containing training data (requires two columns with name and category), 
        batch_size (int) - batch size
        shuflle_pandas (bool) - shuffle the pandas data frame prior to creating batch
        

    Returns:
         Returns a data loader object of batch data
         
    TODO: shuffle data frame
    """
    
    #Open pickle
    input_df = pickle.load(open( input_df, "rb" ) )
    
    #Shuffle pandas
    if shuffle_pandas:
        input_df = input_df.sample(frac=1)
    
    #Shuffle frame
    X = []
    Y= []
    for index, row in input_df.iterrows():
        category_tensor = categoryTensor(row["category"], all_categories, n_categories)
        #category_tensor.size()
        name_tensor = inputTensor(category_tensor=category_tensor, n_letters=n_letters, all_letters_dict=all_letters_dict,line=row["name"]) 
        #name_tensor.size()
        target_tensor=targetTensor(all_letters_dict=all_letters_dict,line=row["name"])
        #line_tensor.size()
        X= X + name_tensor
        Y= Y + target_tensor
    
    X_stacked_tensor = torch.stack(X)
    print("X-size:" + str(X_stacked_tensor.size()))
    Y_stacked_tensor=  torch.stack(Y)
    print("Y-size:" + str(Y_stacked_tensor.size()))
    #Iterate through frame
    train_tensor = data_utils.TensorDataset(X_stacked_tensor, Y_stacked_tensor) 
     
    #Split dataset
    train_loader = data_utils.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = False, drop_last=True)
    print("Number of batches in a epoch:" + str(len(train_loader)))
    
    return train_loader     

def categoryFromOutput(output,all_letters_dict_inverse,probs=False):
    """Get category from output of model

    Returns category_idx and categories
    
    Args:
        output (tensor) - tensor from a model,
        all_letters_dict (dict) - all letters dict,
        probs (bool) -if True, returns probability instead of category_idx
        
    Returns:
         Returns category_idx, categories or probs_num, categories
         
    """
        
    #Convert to probabilities
    if(probs):
        sm = torch.nn.Softmax()
        output = sm(output)
        
    top_n, top_i = output.topk(1) 
    category_i = top_i.squeeze(1)
    category_idx = category_i.tolist()
    categories = []
    for value in category_idx:
        categories.append(all_letters_dict_inverse[value])
    
    if(probs):
        probs_i = top_n.squeeze(1)
        probs_num = probs_i.tolist()
        
        return probs_num, categories
    else:
        return category_idx, categories

def binary_accuracy(preds, y, all_letters_dict_inverse):
    """Get accuracy comparing predictions and target

    Returns accuracy as a number
    
    Args:
        preds (tensor) - tensors of predictions
        y (tensor) - tensor of targets
        
    Returns:
         Returns accuracy 
         
    """
    #Accuracy
    category_idx, category = categoryFromOutput(output=preds, all_letters_dict_inverse=all_letters_dict_inverse)

    #round predictions to the closest integer
    correct = (np.array(category_idx) == np.array(y.tolist())) #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion, all_letters_dict_inverse, print_each_batch_result=False):
    """Train a model for a single epoc and calculate epoch_loss and epoc_accuracy (predicting next character)

    Returns two ints; average epoc_loss and epoch_accuracy
    
    Args:
        model (model.LSTM) - model objje
        iterator (Data Loader) - iterator,
        optimizer (torch.optimizer) - optimizer,
        criterion (NN.criterion) - criterion
        all_letters_dict_inverse (dict) - dictionary of key letters
        
    Returns:
         Returns two ints; average epoc_loss and epoch_accuracy
         
    """    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    hidden = model.initHidden()
    #Init hidden
    
    for input, output in iterator: #iterator:
        #print(input.size())
        input, output = input.to(device), output.to(device)
        
        hidden = (hidden[0].data,hidden[1].data)
        
        #Initilize optimizer
        optimizer.zero_grad()
        
        #Predictions
        predictions, hidden = model(input, hidden)#.squeeze(1)
        
        #Calculate loss
        loss = criterion(predictions.squeeze(0), output.squeeze(1).long()) 
        
        #Calculate accuracy
        acc = binary_accuracy(preds=predictions, y=output.squeeze(1).long(), all_letters_dict_inverse=all_letters_dict_inverse)
        
        #Backward propagation
        loss.backward(retain_graph=True)
        
        #Optimizer
        optimizer.step()
        
        #Calculate loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        if print_each_batch_result:
            print(loss.item(), acc.item())
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def store_model(model,
               optimizer,
               model_folder,
               model_name,
               model_desc,
               all_letters, 
               all_letters_dict,
               all_letters_dict_inverse, 
               n_letters,
               category_lines ,
               all_categories, 
               all_categories_dict , 
               n_categories,
               train_loader,
               input_df,
               n_epochs,
               batch_size,
               n_layers,
               hidden_size,
               dropout_fact,
               learning_rate,
               model_type,
               avg_loss,
               avg_accuracy,
               training_names,
               ):
    
    
    
    """Stores the model

    Args:
        all parameters used for training including the model
        
    Returns:
         Nothing
         
    """       
    
    
    print("Saving the model...")
    print(model_name)
    #torch.save(model.state_dict(), model_folder+model_name+ '.pt')
    #torch.save(optimizer.state_dict(), model_folder+model_name+ '.opt')
    #torch.save(model, model_folder+model_name+ '.rnn')
    
    #Save complete set and all supporting data sets
    torch.save({
        'model_desc':model_desc,
        'model':model,
        'optimizer':optimizer,
        'all_letters': all_letters, 
        'all_letters_dict': all_letters_dict,
        'all_letters_dict_inverse':all_letters_dict_inverse , 
        'n_letters':n_letters,
        'category_lines': category_lines ,
        'all_categories':all_categories, 
        'all_categories_dict':all_categories_dict , 
        'n_categories':n_categories,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': optimizer.state_dict(),
        'train_loader':train_loader,
        'input_df':input_df,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'n_layers': n_layers,
        'hidden_size': hidden_size,
        'dropput_fact': dropout_fact,
        'learning_rate':learning_rate,
        'model_type': model_type,
        'avg_loss':avg_loss,
        'avg_accuracy':avg_accuracy,
        'training_names':training_names
        
        }, model_folder+model_name+'.whl')
    
    #Save only model parameters and supporting data
    torch.save({
        'model_desc':model_desc,
        'all_letters': all_letters, 
        'all_letters_dict': all_letters_dict,
        'all_letters_dict_inverse':all_letters_dict_inverse , 
        'n_letters':n_letters,
        'category_lines': category_lines ,
        'all_categories':all_categories, 
        'all_categories_dict':all_categories_dict , 
        'n_categories':n_categories,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': optimizer.state_dict(),
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'n_layers': n_layers,
        'hidden_size': hidden_size,
        'dropput_fact': dropout_fact,
        'learning_rate':learning_rate,
        'model_type': model_type,
        'avg_loss':avg_loss,
        'avg_accuracy':avg_accuracy,
        'training_names':training_names
        
        }, model_folder+model_name+'.pt')
    
def word_similarity(word_to_compare='Vignir',
                    list_of_words=["Heigigr","Beðurni"],
                    return_top_n=20,
                    use_cut_off=False,
                    cut_off = 0.5,
                    sim_measure='Levenshtein' ,#SequenceMatcher #Jaro-Winkler #Hamming,
                    min_characters=2, #Null for no restriction,
                    filter_non_capital_letters = True
                    ):
    
    """Compare similarity between a word and a list of words

    Returns list of similar words/names based on a similarity measure
    
    Args:
        word_to_compare (str) -word to compare with each value in list
        list_of_words (lst) - list of strings to compare against
        return_top_n (int) - return only top n 10 results based on similarity measure
        use_cut_off (bool) - whether to use a cut off value based on similarity
        cut_off (int) - cut off value
        
    Returns:
         Returns two ints; average epoc_loss and epoch_accuracy
         
    """       
    word_similarity_list=[]
    for word in list_of_words:
        dict_Words ={}
        dict_Words['word_to_compare']=word_to_compare
        dict_Words['word_to_compare_against']=word
        if sim_measure=='Levenshtein':
            ##dict_Words['similarity']=Levenshtein.ratio(word_to_compare, word)
            dict_Words['similarity']=jellyfish.levenshtein_distance(word_to_compare, word)*-1
            dict_Words['similarity_measure']='Levenshtein'
        elif sim_measure=='SequenceMatcher':
            dict_Words['similarity']=SequenceMatcher(None,word_to_compare, word).ratio()
            dict_Words['similarity_measure']='SequenceMatcher'
            #https://docs.python.org/2.4/lib/sequencematcher-examples.html
        elif sim_measure=='Jaro-Winkler':
            dict_Words['similarity']=jellyfish.jaro_winkler(word_to_compare, word)
            dict_Words['similarity_measure']='Jaro-Winkler'
        elif sim_measure=='Hamming':
            dict_Words['similarity']=jellyfish.hamming_distance(word_to_compare, word)*-1
            dict_Words['similarity_measure']='Hamming'
        word_similarity_list.append(dict_Words)
        
    #Convert to frame
    df_word_similarity = pd.DataFrame(word_similarity_list)
   
    #Sort
    df_word_similarity=df_word_similarity.sort_values(by='similarity', ascending=False)
    
    #Return top results
    if return_top_n>0:
        if len(df_word_similarity)>return_top_n:
            df_word_similarity=df_word_similarity[0:return_top_n]
    else:
        return df_word_similarity[0:0]
    
    #Whether to use cutoff        
    if use_cut_off:
        df_word_similarity=df_word_similarity[df_word_similarity.similarity>cut_off]
    
    #Filter min characters
    if min_characters>0:
        df_word_similarity=df_word_similarity[df_word_similarity.word_to_compare_against.str.len()>min_characters]
        
    #Filter out words that does not start with a large character
    if filter_non_capital_letters:
        df_word_similarity=df_word_similarity[df_word_similarity.word_to_compare_against.str.istitle()]
    
    return df_word_similarity 


def load_model(file_name='../model/' + 'NORWEGIAN_FEMALE_NAMES_LSTM_NLAYERS_2_HIDDENSIZE_128_N_EPOCS_500_DEVICE_CUDA.pt', load_optimizer=False):
    """Returns a model

    Returns a model and necessary functions for scoring
         
    """      
    #Load model
    checkpoint = torch.load(file_name, map_location='cpu')

    #Network
    INPUT_DIM = checkpoint['n_letters']+checkpoint['n_categories']
    OUTPUT_DIM = checkpoint['n_letters']
    HIDDEN_DIM = checkpoint['hidden_size']
    N_LAYERS = checkpoint['n_layers']
    DROPOUT = checkpoint['dropput_fact']
    
    all_letters_dict=checkpoint['all_letters_dict'] 
    all_letters_dict_inverse=checkpoint['all_letters_dict_inverse']
    all_categories=checkpoint['all_categories']
    category_lines=checkpoint['category_lines']
    n_categories=checkpoint['n_categories']
    n_letters=checkpoint['n_letters']
    training_names=checkpoint['training_names']
    
    #Define model and load dictionaries
    model = LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, torch.device('cpu' ),N_LAYERS, DROPOUT)
    model.load_state_dict(checkpoint['model_state_dict'])
    if  load_optimizer:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        return model, optimizer, all_letters_dict, all_letters_dict_inverse, all_categories,category_lines,n_categories,n_letters, training_names
    else:
        return model, all_letters_dict, all_letters_dict_inverse, all_categories,category_lines,n_categories,n_letters, training_names
    


#TODO: Develop evauluate function
#def evaluate(model_char, iterator, criterion):
#    
#    epoch_loss = 0
#    epoch_acc = 0
#    
#    model_char.eval()
#    
#    with torch.no_grad():
#    
#        for input, output in iterator:
#            
#            input, output = input.to(device), output.to(device)
#            
#            predictions = model_char(input.transpose(0,1))
#            
#            loss = criterion(predictions.squeeze(0), output.squeeze(1).long()) 
#            
#            acc = binary_accuracy(predictions, output.squeeze(1).long())
#
#            epoch_loss += loss.item()
#            epoch_acc += acc.item()
#        
#    return epoch_loss / len(iterator), epoch_acc / len(iterator)
