# RNN Generator PyTorch

This is a RNN Generator developed in [PyTorch](http://pytorch.org/). The code can be a run as stand-alone command line tool as well. 

Given a one-column pickled pandas data frame the code can train a model and spit out new occurrences/variances of a name that the model has not seen before. 

Example of a pickled panda with known rap names could be:

input_df:

| rap_names|      
|----------|
| 21 Savage|  
| Big Boi  |
|Lil Dicky |
| ........ | 

The model then can output new rap names from a trained model and given an input of a temperature variable:

| model_rap_names	|      
|----------						|
| Drap Gott Wuss		|  
| Hip Peest  				|
| Quastin 					|
| ........ 							|

There are also collected pickled pandas the model can train on: 

|Name            					|Description                         			|Rows                       |
|----------------			 	|---------------------------	|-----------------------------|
|data/df_data.pickle	 	|`'Norwegian business names from enhetsregisteret'`            	|1 089 313          |
|data/blackspeech.pickle    				|`"Mordor names from LOTR"`            	|1000ish            |
|data/icelandic_male_names.pickle      					|`Islandic male names`|1951|
|data/rorlegger.pickle     					|`Norwegian plumber companies names`|10000ish|

# Files

1. The data is stored in the file folder
2. The models are stored in the model folder
3. All the code in the code folder

# Running the code

The code can be run through command line or in some other Python IDE. 