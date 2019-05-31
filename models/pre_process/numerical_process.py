import tensorflow as tf
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split

def unload_data(data_path,train_input_file,train_output_file,test_input_file,test_output_file):
    inputs = []
    outputs = []
    with open(data_path,'r') as f:
        data = json.load(f)
    data = [cleaned_data for cleaned_data in data if cleaned_data['volume'][0] != '-']
    for index in range(len(data)):
        if len(data) - index > 11:
            input_buffer = []
            for row in data[index+1:index+11]:
                temp_buffer = []
                temp_buffer.append(float(row['open'][0]))
                temp_buffer.append(float(row['close'][0]))
                temp_buffer.append(float(row['high'][0]))
                temp_buffer.append(float(row['low'][0]))
                temp_buffer.append(float(row['volume'][0].replace(',','')))
                temp_buffer.append(float(row['mark_cap'][0].replace(',','')))
                input_buffer.append(temp_buffer)
            inputs.append(input_buffer)
            outputs.append(data[index]['close'][0])
    inputs = np.asarray(inputs) 
    outputs = np.asarray(outputs) 

    print('input size: {},output size: {}'.format(len(inputs),len(outputs)))
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.15)
        
    np.save(train_input_file,x_train)
    np.save(train_output_file,y_train)
    np.save(test_input_file,x_test)
    np.save(test_output_file,y_test)
    

def input_fn(input_path,output_path,params):
    
    inputs = np.load(input_path)
    outputs = np.load(output_path)

    print(inputs)

    assert inputs.shape[0] == outputs.shape[0]

    with tf.device('/cpu:0'):
        dataset = (tf.data.Dataset.from_tensor_slices((inputs,outputs))
                .shuffle(inputs.shape[0])
                .batch(params['batch_size'])
                .prefetch(params['buffer_size'])
                )

    return dataset

#unload_data('data/historical_data.json','data/train_inputs.npy','data/train_outputs.npy',
#        'data/test_inputs.npy','data/test_outpus.npy')
data = input_fn('data/train_inputs.npy','data/train_outputs.npy',{'batch_size':32,'buffer_size':1})
