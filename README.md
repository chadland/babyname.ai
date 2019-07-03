
![alt text](https://raw.githubusercontent.com/chadland/babyname.ai/Html/images/logo.PNG/images/logo.PNG)

# Babyname.ai

AI service to generate creative baby names
This is a RNN Generator developed in [PyTorch](http://pytorch.org/). The code can be a run as stand-alone command line tool as well. 

The solution has been puplished in AWS: [Babyname.ai](https://d14rhc8ldmadon.cloudfront.net/)

The soltution will take a first name as input and the RNN model will then generate names and the most similar ones are returned as output. In the solution the AI names will be mixed with actual names based on a creativity parameter ranging from 0-100. The higher this value is, the more AI names will be posted. 

# Folders

1. awslambda: contains code that the current AWS Lambda model is built upon
2. code: python code for the solution, to train and test models
3. data: data sets to train on
4. documentation: extended documentation material
5. Html: front end html solution
6. model: various models for the solution

# Running the code

The code can be run through command line or in some other Python IDE. 

# Models 

The name generating models are trained for specific geneders from specific countries. Feel free to add new data sets and models. If they are of good quality I would be happy to add them to the AWS-solution. 
