import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def unload_data(data_path,input_path,output_path):
    inputs = []
    outputs = []

    df = pd.read_csv(data_path).values
    for row in df:
        print(df)
    '''
    print 'input size: {},output size: {},vocab size: {}'.format(len(inputs),len(outputs),len(unique))

    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.15)
  
    for counter,input_,output in enumerate(zip(x_train,y_train)):
        with open(input_path+'_train_'+str(counter)+'.npy','w') as f:
            np.save(f,input_)
        with open(output_path+'_train_'+str(counter)+'.npy','w') as f:
            np.save(f,output)
        train_input_paths.append('data/inputs_train_'+str(counter)+'.npy')
        train_output_paths.append('data/outputs_train_'+str(counter)+'.npy')

    for counter,input_,output in enumerate(zip(x_test,y_test)):
        with open(input_path+'_test_'+str(counter)+'.npy','w') as f:
            np.save(f,input_)
        with open(output_path+'_test_'+str(counter)+'.npy','w') as f:
            np.save(f,output)
        test_input_paths.append('data/inputs_test_'+str(counter)+'.npy')
        test_output_paths.append('data/outputs_test_'+str(counter)+'.npy')


    with open(train_input_path_storage, 'a') as f:
        for i in train_input_paths:
            f.write(i+'\n')

    with open(train_output_path_storage, 'a') as f:
        for o in train_output_paths:
            f.write(o+'\n')
   
    with open(test_input_path_storage, 'a') as f:
        for i in test_input_paths:
            f.write(i+'\n')

    with open(test_output_path_storage, 'a') as f:
        for o in test_output_paths:
            f.write(o+'\n')
   
    with open(vocab_path, 'a') as f:
        for v in unique:
            f.write(v+'\n')
    '''

def read_npy_file(x,y):
    inp = np.load(x.numpy())
    output = np.load(y.numpy())
    return inp.astype(np.float32),output.astype(np.float32)

def input_fn(input_path,output_path,params):
    num_elements = sum(1 for line in open(input_path))
    num_labels = sum(1 for line in open(output_path))

    assert num_elements == num_labels

    input_paths = []
    output_paths = []

    with open(input_path) as f:
        for line in f:
            input_paths.append(line[:-1])
    with open(output_path) as f:
        for line in f:
            output_paths.append(line[:-1])
   
    input_paths = input_paths[:10000]
    output_paths = output_paths[:10000]

    with tf.device('/cpu:0'):
        dataset = (tf.data.Dataset.from_tensor_slices((input_paths,output_paths))
                .map(lambda x,y: tuple(tf.py_function(read_npy_file, [x,y], [tf.float32,tf.float32])),num_parallel_calls=3)
                .shuffle(num_elements)
                .batch(params['batch_size'])
                .prefetch(params['buffer_size'])
                )

    return dataset

unload_data('data/bitcoin_price.csv',3,4)
#data = input_fn('../data/train_input_paths.txt','../data/train_output_paths.txt',{'batch_size':1,'buffer_size':10})
