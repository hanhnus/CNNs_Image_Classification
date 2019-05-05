#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

"""
#from tensorflow.keras.models import Sequential

from datetime                import datetime
from time                    import time, localtime, strftime
from keras.backend           import clear_session
from keras.models            import Sequential
from keras.utils             import np_utils
from keras.layers            import Dense, Convolution2D, MaxPooling2D, Flatten, Activation
from keras.optimizers        import Adam
from keras.layers            import Dropout
from PIL                     import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelBinarizer   # for 1-hot
from tensorflow.keras        import backend as K
from tensorflow.keras.preprocessing.image import img_to_array
import pandas     as pd
import numpy      as np
import tensorflow as tf
import random
import os
import re
import gc
import sys

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)
FMT  = '%Y-%m-%d %H:%M:%S.%f'
train_test_split_test_size = 0.3

# Read images
def read_images():       
#    directory = '../Train_data/all/'
#    directory = '../Train_data/all12/'
#    directory = '../Train_data/1500/'
#    directory = '../Train_data/300/'
    directory = '../Train_data/all_-1/'
    count = np.zeros(100)    # record the count of images with certain last 2 digit (00 - 99)
    '''
    read all the images in the folder into memory, from the images with the last 2 digit of 00 to that of 99
    100 image array variables (dynamic) & 100 label variables (dynamic) were defined, to speed up the image data reading process
    eg.
    ----------------------------------------------------------
       var    |                    stores
    X_list_00 | all the images out of 4500 ending with *00.jpg
    X_list_01 | all the images out of 4500 ending with *01.jpg
    ...       | ...
    X_list_99 | all the images out of 4500 ending with *99.jpg
    y_list_00 | all the labels ending with *00.jpg
    y_list_01 | all the labels ending with *01.jpg
    ...       | ...
    y_list_99 | all the labels ending with *99.jpg
    ----------------------------------------------------------
    output: 200 vectors

    '''
    for imgname in os.listdir(directory):
        img       = Image.open(directory + imgname)
        img_array = img_to_array(img)                             # keras
        #print(img_array.shape)
        for last_2digit in range(100):
            x = str(last_2digit).zfill(2) + '.jpg'
            if x in imgname:
                count[last_2digit] += 1
                if count[last_2digit] == 1:
                    locals()['X_list_'+str(last_2digit).zfill(2)] = np.array([img_array])
                    locals()['y_list_'+str(last_2digit).zfill(2)] = np.array([''.join(re.findall('([^ ]*)_', imgname))])
                else:
                    locals()['X_list_'+str(last_2digit).zfill(2)] = np.append(locals()['X_list_'+str(last_2digit).zfill(2)], img_array)
                    locals()['y_list_'+str(last_2digit).zfill(2)] = np.append(locals()['y_list_'+str(last_2digit).zfill(2)], ''.join(re.findall('([^ ]*)_', imgname)))
                print('{a}\t\t{b}\t{c}'.format(a = imgname, b = locals().get('X_list_'+str(last_2digit).zfill(2)).shape, c = locals().get('y_list_'+str(last_2digit).zfill(2)).shape))
    # Garbage Collection
    del img
    del img_array
    del x
    del directory
    gc.collect() 
      

    '''
	1. reshape the vectors to arrays with shape of (300, 300, 3)
	2. normalisation (by / 255)
	try: in case there is no certain last-2-digit
    '''
    for last_2digit in range(100):
        try:
            locals()['X_'+str(last_2digit).zfill(2)] = locals()['X_list_'+str(last_2digit).zfill(2)].reshape(-1, 300, 300, 3)
            locals()['X_'+str(last_2digit).zfill(2)] = locals()['X_'+str(last_2digit).zfill(2)].astype('float32')
            locals()['X_'+str(last_2digit).zfill(2)] = locals()['X_'+str(last_2digit).zfill(2)] / 255.0            
            print(str(last_2digit).zfill(2))
            print(locals().get('X_'+str(last_2digit).zfill(2)).shape)
            del locals()['X_list_'+str(last_2digit).zfill(2)]
            gc.collect()
        except:
            pass

    '''
    for small amount of examples,
    to find the smallest last-2-digit in the folder
    '''
    for j in range(100): 
        if count[j] > 0:
            min_last_two_digit = j            
            break
    print(min_last_two_digit)
    print('X_' + str(min_last_two_digit).zfill(2))
    # initialise X and y by reading the dynamic X, y with the smallest last-2-digit
    X = locals().get('X_'+str(min_last_two_digit).zfill(2))
    y = locals().get('y_list_'+str(min_last_two_digit).zfill(2))
    # concatenate all the rest dynamic X, y to X, y
    for last_2digit in range(min_last_two_digit + 1, 100):
        try:
            print(last_2digit)
            X = np.concatenate((X, locals()['X_'+str(last_2digit).zfill(2)]))
            y = np.concatenate((y, locals()['y_list_'+str(last_2digit).zfill(2)]))
            print(X.shape)
            del locals()['X_'+str(last_2digit).zfill(2)]
            del locals()['y_list_'+str(last_2digit).zfill(2)]
            gc.collect()   
        except:
            pass

    print(dir())
    print(sys.getsizeof(X))
    print(sys.getsizeof(count))
    print(sys.getsizeof(imgname))
    print(sys.getsizeof(j))
    print(sys.getsizeof(last_2digit))
    print(sys.getsizeof(min_last_two_digit))
    print(sys.getsizeof(y))

    # EDA
    print('Max:    ',X.max()) 
    print('Min:    ',X.min())
    print('Mean:   ',X.mean())  
    print('STD:    ',X.std())
    print('Median:',np.median(X))
          



    print(count)
    print('shape of X:                ', X.shape)
    print('shape of the first image:  ', X[0].shape)
    '''
    test by printing out the first 10 images one by one
    '''
#    for a in range(10):
#        new_im1 = Image.fromarray(X[a].astype('uint8'))
#        new_im1.show()
    print(y)

    # generate 1-hot array
    encoder = LabelBinarizer()
    y_1hot  = encoder.fit_transform(y)
    print(y_1hot)
    del y
    gc.collect()

    # generate train split and test split
    X_train, X_test, y_train, y_test = train_test_split(
    	X, 
    	y_1hot, 
    	test_size = train_test_split_test_size, 
    	shuffle   = True)

    del X
    del y_1hot
    gc.collect()

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(type(X_train))
    print(X_train)


    print(dir())
    print(sys.getsizeof(X_test))
    print(sys.getsizeof(X_train))
    print(sys.getsizeof(encoder))
    print(sys.getsizeof(y_test))
    print(sys.getsizeof(y_train))
    return X_train, X_test, y_train, y_test

def construct_model(X_train, X_test, y_train, y_test):
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential()
    model.add(Convolution2D(
        input_shape = (300, 300, 3),
        filters     = 64,    # 64 > 32 > 16
        kernel_size = 5,     # 5 > 3
        strides     = 3,     # 3 > 2 > 1
        padding     = 'same',
        data_format = 'channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size   = 4,     # 4 > 3 > 2 > 1
        strides     = 1,     # 1 > 2
        padding     = 'same', 
        data_format = 'channels_last'))
    model.add(Convolution2D(
        16,                  # Conv2filter 16 > 8 
        7,                   # Conv2kernel 7 > 5 > 3
        strides     = 5,     # 5 > 4 > 3 > 2 > 1 
        padding     = 'same', 
        data_format = 'channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        6,                   # Pool2size 6 > 5 > 4 > 3 > 2 > 1 
        2,                   # Pool2stride 2 > 1
        padding     = 'same', 
        data_format = 'channels_last'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(
        optimizer   = Adam(lr = 1e-4), 
        loss        = 'categorical_crossentropy',
        metrics     = ['accuracy'])
    return model

def train_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # Add your code here
    hist = model.fit(
    	X_train, 
    	y_train, 
    	epochs          = 100, 
    	validation_data = (X_test, y_test), 
    	shuffle         = True)
    loss_train, accuracy_train = model.evaluate(X_train, y_train)
    loss_test,  accuracy_test  = model.evaluate(X_test,  y_test)
    print('Training Set Loss:    ', loss_train)
    print('Training Set Accuracy:', accuracy_train)
    print('Test Set Loss:        ', loss_test)
    print('Test Set Accuracy:    ', accuracy_test)    
    return model, loss_train, accuracy_train, loss_test, accuracy_test, hist

def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model.save("model/model.h5")
    print("Model Saved Successfully.")

def Save_training_record_to_csv():
    current_time   = strftime('%Y-%m-%d_%H-%M-%S', localtime())
    temp_file_name = "CNN_Training_Record_@" + str(current_time) + ".csv"
    df.to_csv("CNN_Training_Record.csv", encoding = "utf_8_sig")
    df.to_csv(temp_file_name, encoding = "utf_8_sig")

def Save_modeling_process_to_csv():
    current_time   = strftime('%Y-%m-%d_%H-%M-%S', localtime())
    temp_file_name = "CNN_Model_Detail_@" + str(current_time) + ".csv"
    df_history.to_csv(temp_file_name, encoding = "utf_8_sig")



if __name__ == '__main__':
    i = 0
    csv_Table_Title = ['loss_train', 'accuracy_train', 'loss_test', 'accuracy_test']
    df = pd.DataFrame(columns = csv_Table_Title)
    X_train, X_test, y_train, y_test = read_images()
    model = construct_model(X_train, X_test, y_train, y_test)
    model, loss_train, accuracy_train, loss_test, accuracy_test, hist = train_model(model)
    df_history = pd.DataFrame(hist.history)   # save loss, acc, val_loss, val_acc during the whole training process
    print(df_history)
    #clear_session()
    df.loc[i] = [loss_train, accuracy_train, loss_test, accuracy_test]
    Save_training_record_to_csv()
    Save_modeling_process_to_csv()
    save_model(model)
