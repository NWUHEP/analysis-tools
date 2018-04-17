#/opt/anaconda3/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import seaborn as sns
from root_pandas import read_root


if __name__ == '__main__':
# get the data
    feature_names = [
                     'pt', 'eta', 'phi', 'ptErr',
                     #'trkIso', 'ecalIso', 'hcalIso',
                     'chHadIso', 'gammaIso', 'neuHadIso', 'puIso',
                     #'puppiChHadIso', 'puppiGammaIso', 'puppiNeuHadIso',
                     #'puppiChHadIsoNoLep', 'puppiGammaIsoNoLep', 'puppiNeuHadIsoNoLep',
                     'd0', 'dz', 'sip3d',
                     'tkNchi2', 'muNchi2', 'trkKink', 'glbKink',
                     'trkHitFrac', 'chi2LocPos', 'segComp', 'caloComp',
                     #'q',
                     'nValidHits', 'nTkHits', 'nPixHits', 'nTkLayers', 'nPixLayers', 'nMatchStn',
                     #'typeBits', 'selectorBits', 'pogIDBits', # this will need to be decoded into onehots
                    ]
    target_names = [ 'genPt', 'genEta', 'genPhi', 'genMatched', ]
    cuts = '3. < pt < 30. \
            and abs(d0) < 0.5 and abs(dz) < 1.0 and abs(sip3d) < 100. \
            and 0. < muNchi2 < 10. and 0. < tkNchi2 < 200. \
            and trkKink < 999. and glbKink < 500000. \
            and chi2LocPos < 2000.'


# get the dataframes (this will take a long time)
    input_trees = ['ttbar']
    df_test = 0
    for df in read_root('data/output_ttbar.root', key='muons_tree_ttbar_semilep', chunksize=10000000):
        df = df[feature_names + target_names].query(cuts)
        neutral_iso = df.gammaIso + df.neuHadIso - 0.5*df.puIso
        neutral_iso[neutral_iso < 0.] = 0.
        
        # add combined pf isolation for comparison to official recommendation
        df['pf_combined'] = (df.chHadIso + neutral_iso) / df.pt
        
        # take the absolute value of variables that are symmetric
        df['eta'] = np.abs(df['eta'])
        df['d0'] = np.abs(df['d0'])
        df['dz'] = np.abs(df['dz'])
        df['sip3d'] = np.abs(df['sip3d'])
        
        df_test = df
        break

    # make some plots
    trunc_features = ['pt', 'eta', 'phi', 'chHadIso', 'gammaIso', 'neuHadIso', 'puIso']
    df = df_test[:1000][trunc_features + ['genMatched']]

    g = sns.PairGrid(df, vars=trunc_features, hue='genMatched')
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    plt.show()

    # data preprocessing
    targets  = df_test['genMatched'].values

    features = df_test[feature_names].values
    # scale features; convert type bits to one hots
    mm_scaler = preprocessing.MinMaxScaler()
    features  = mm_scaler.fit_transform(features)

    df_scaled = pd.DataFrame(features, columns=feature_names)
    df_scaled['genMatched'] = targets

    # randomize data; split into testing and training sets
    ix = np.arange(features.shape[0])
    np.random.shuffle(ix)
    ix_split = int(0.8*ix.size)
    x_train = features[ix,][:ix_split]
    x_test  = features[ix,][ix_split:]
    y_train = targets[ix] #to_categorical(targets[ix])
    y_test  = y_train[ix_split:]
    y_train = y_train[:ix_split]


    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, feature in enumerate(feature_names):
        a_true = df_scaled[df_scaled['genMatched']][feature]
        a_fake = df_scaled[~df_scaled['genMatched']][feature]
        axes[int(i/5)][i%5].hist([a_true, a_fake], bins=50, stacked=True, color=['C0', 'C1'])
        axes[int(i/5)][i%5].set_title(feature)
        axes[int(i/5)][i%5].set_yscale('log')
    plt.show()

    # define the network architecture:
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=features.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['binary_accuracy'])

    # train the model
    model.fit(x_train, y_train,
              epochs = 20,
              batch_size = 128,
              validation_split = 0.1)

    # get prediciton, calculate efficiencies
    p = model.predict(x_test).flatten()
    p_fake = p[y_test==0]
    p_real = p[y_test==1]
    eff_fake = np.array([p_fake[p_fake > cut].size for cut in np.arange(0, 1, 0.01)])/p_fake.size
    eff_real = np.array([p_real[p_real > cut].size for cut in np.arange(0, 1, 0.01)])/p_real.size

    # calculate efficiencies from isolation
    iso_real = df_test[df_test.genMatched].pf_combined
    iso_fake = df_test[~df_test.genMatched].pf_combined
    eff_iso_real = np.array([iso_real[iso_real < cut].size for cut in np.arange(0, 10, 0.01)])/iso_real.size
    eff_iso_fake = np.array([iso_fake[iso_fake < cut].size for cut in np.arange(0, 10, 0.01)])/iso_fake.size

    # make some performance plots
    fig, axes = plt.subplots(1, 2, sharey=False, figsize=(12,6), facecolor='white')
    h1 = axes[0].hist(p_fake, bins=50, histtype='stepfilled', alpha=0.5)
    h2 = axes[0].hist(p_real, bins=50, histtype='step', linewidth=2.)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('DNN output')
    axes[0].set_ylabel('Entries / bin')
    axes[0].legend(['fake muons', 'real muons'])

    axes[1].plot(eff_real, 1. - eff_fake)
    axes[1].plot(eff_iso_real, 1 - eff_iso_fake, c='C2')
    axes[1].set_xlabel('muon efficiency')
    axes[1].set_ylabel('muon purity')
    axes[1].legend(['DNN', 'pf iso/pt'])

    plt.show()

