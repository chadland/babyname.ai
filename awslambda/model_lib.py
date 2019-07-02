# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:02:10 2019

@author: c202acox
"""

import torch
import torch.nn as nn

train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device,n_layers=2,dropout_fact=0.5, batch_size=64):
        super(LSTM, self).__init__()
        self.n_layers=n_layers
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.dropout = nn.Dropout(dropout_fact)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden):
        r_output,hidden = self.lstm(input.unsqueeze(0).float() ,hidden)  
        #r_output,hidden = lstm(input.unsqueeze(0).float() ,hidden)  
        #r_output.size()
        out = self.dropout(r_output)
        #out = dropout(r_output)
        
        out = out.contiguous().view(-1, self.hidden_size)
        #out = out.contiguous().view(-1, hidden_size)
        
        out = self.fc(out)
        #out = fc(out)
        #out.size()
        
        out=self.softmax(out)
        #out=softmax(out)
        #out.size()
        
        return out, hidden

    def initHidden(self):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        #weight = next(self.parameters()).data
        #hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
        return (torch.zeros(self.n_layers, self.batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.n_layers, self.batch_size, self.hidden_size, device=self.device))
        
    def initHiddenPrediction(self):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        #weight = next(self.parameters()).data
        #hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
        return (torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device ),
                torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device))
