'''
Created on Dec 7, 2017

@author: go
'''

import keras
from keras.models import Sequential
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import sgd
from keras.datasets import mnist
# from keras.utils import plot_model

from gabor_init import gabor_init

# dataset related parameters
input_shape = (28,28,1)
num_classes = 10
batch_size = 32
epochs = 1

def load_data():
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # reshape dataset to feed it into model properly
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # normalize train/test data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    
    # convert class vectors to matrices as binary
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print('Number of train samples in MNIST: ', x_train.shape[0])
    print('Number of test samples in MNIST: ', x_test.shape[0])
    
    return (x_train, y_train), (x_test, y_test)

# Build AlexNet-like model
def build_model():
    model = Sequential()
    
    # Convolution layer 1
    model.add(Conv2D(96, kernel_size=(5,5), padding='same',
                     kernel_initializer=gabor_init, 
                     bias_initializer='zeros', 
                     input_shape=input_shape))
    model.add(Activation('relu'))
              
    # Convolution layer 2
    model.add(Conv2D(96, kernel_size=(1,1), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
              
    # Convolution layer 3
    model.add(Conv2D(192, kernel_size=(5,5), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
              
    # Convolution layer 4
    model.add(Conv2D(192, kernel_size=(1,1), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    
    # Convolution layer 5
    model.add(Conv2D(192, kernel_size=(3,3), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(192, kernel_size=(1,1), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(10, kernel_size=(1,1), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    
    model.add(AveragePooling2D(pool_size=(6,6)))
              
    model.add(Flatten())
    
    # Dense layer 3 (fc8)
    model.add(Dense(num_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('softmax'))
    
    return model

def main():
    # load dataset
    print('Loading dataset...')
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # build model
    print('Building model...')
    model = build_model()

    #plot_model(model, to_file='model.png')
    #exit()    
    # compile model
    print('Compiling model...')
    optimizer = sgd(0.01, 0.9, 0.0005)
    #optimizer = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # train model
    for i in range(20):
        print(i+1)
        print('Training model...')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    
        # evaluate model
        print('Evaluating model...')
        score = model.evaluate(x_test, y_test)
        print('Test accuracy: ', score[1])
        print('Test loss: ', score[0])

if __name__ == '__main__':
    main()
