

import numpy as np
import pandas as pd

from sklearn import preprocessing 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


def preprocess_data(df, cfeatures, dfeatures, bounds_dict):
    '''
    Carries out preprocessing steps for datasets used for NN based analysis.
    '''

    # remove variables that are not being used
    df = df[cfeatures + dfeatures + ['label', 'weight']]

    # combine high jet multiplicity bins
    df.loc[df.n_jets > 4, 'n_jets']       = 4
    df.loc[df.n_bjets > 2, 'n_bjets']     = 2
    df.loc[df.n_fwdjets > 1, 'n_fwdjets'] = 1

    for f in cfeatures:
        if f in bounds_dict.keys():
            xmin, xmax = bounds_dict[f]
            df.loc[df[f] > xmax, f] = xmax
            df.loc[df[f] < xmin, f] = xmin

    # scale features; convert type bits to one hots
    mm_scaler = preprocessing.MinMaxScaler()
    df.loc[:,cfeatures]  = mm_scaler.fit_transform(df[cfeatures])

    # convert discrete valued variable to binary vectors
    df = pd.get_dummies(df, columns=dfeatures)

    return df

def initialize_model(input_size, output_size):
    '''
    Returns an initialized keras NN
    '''

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_size))
    #model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['categorical_accuracy'])#, 'fmeasure', 'precision', 'recall'])  
    #model.compile(loss='binary_crossentropy',
    #              optimizer=sgd,
    #              metrics=['binary_accuracy'])  
    return model

