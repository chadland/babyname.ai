import os
os.environ['ACCESS_KEY'] = 'AKIAJCHRSBNIQIC2735A'
os.environ['SECRET_KEY'] = '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'
os.environ['MODEL_KEY'] = 'model.pth'
os.environ['PARAMS_KEY'] = 'params.pkl'
os.environ['BUCKET_NAME'] = 'babyname.ai'
os.environ['model_tmp'] = '/tmp/' #'/tmp/ in deployment

event = {'first_name':'Christer', 
		'name_creativity':'0.5',
         'model_name':'models/NORWEGIAN_FEMALE_NAMES_LSTM_NLAYERS_2_HIDDENSIZE_128_N_EPOCS_500_DEVICE_CUDA.pt'
         }

from lambda_function import lambda_handler
lambda_handler(event=event, context=None)