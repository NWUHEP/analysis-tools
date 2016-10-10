from __future__ import division

import sys
from timeit import default_timer as timer
from itertools import product

import numpy as np
from numpy.polynomial.legendre import legval
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

import nllfitter.fit_tools as ft
import nllfitter.plot_tools as pt

# global options
np.set_printoptions(precision=3.)

def dalitz_plot(data):
    df = data.query('dilepton_b_mass < 300 and dilepton_j_mass < 300')
    s1 = df['dilepton_b_mass']**2
    s2 = df['dilepton_j_mass']**2
    sns.jointplot(x=s1, y=s2, kind='hex', color='b', xlim=[0, 90000], ylim=[0,90000])

def pair_plot(df_data):
    '''
    ### work on this ###
    df_data['test'] = ['1b1f' if x == 1 else '1b1c' for x in df_data.n_fwdjets]
    df_data['test'] = np.where(((df_data['dilepton_mass'] > 24) & (df_data['dilepton_mass'] < 32)), df_data['test']+'_sr', 'sideband')
    g = sns.pairplot(df_data, 
                 vars      = ['dilepton_mass', 'dilepton_b_mass', 'dilepton_pt'],
                 hue       = 'test',
                 palette   = 'husl',
                 kind      = 'scatter',
                 diag_kind = 'hist',
                 markers   = ['o', 's', 'D'],
                 plot_kws  = dict(s=50, linewidth=0.5),
                 diag_kws  = dict(bins=30, histtype='stepfilled', stacked=True, alpha=0.5, linewidth=1),
                 size=3, aspect=2,
                )
    g.savefig('plots/pairplot.png')
    plt.close()
    '''
    pass

if __name__ == '__main__':

    ### Start the timer
    start = timer()

    ### Configuration
    pt.set_new_tdr()
    ntuple_dir  = 'data/flatuples/mumu_2012'
    selection   = ('mumu', 'combined')
    period      = 2012
    output_path = 'plots/fits/{0}_{1}'.format('_'.join(selection), period)

    datasets    = [
                   #'muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D', 
                   #'electron_2012A', 'electron_2012B', 'electron_2012C', 'electron_2012D', 
                   'ttbar_lep', 'ttbar_semilep',
                   'zjets_m-50', 'zjets_m-10to50',
                   #'t_s', 't_t', 't_tw', 'tbar_s', 'tbar_t', 'tbar_tw', 
                   #'ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q', 'zz_2l2nu',
                   'bprime_xb'
                  ]
    features = [
                #'run_number', 'event_number', 'lumi', 'weight',
                'lepton1_pt', 'lepton1_eta', 'lepton1_phi', 
                #'lepton1_iso', 'lepton1_q', 'lepton1_flavor', 'lepton1_trigger',
                'lepton2_pt', 'lepton2_eta', 'lepton2_phi',  
                #'lepton2_iso', 'lepton2_q', 'lepton2_flavor', 'lepton2_trigger',
                'lepton_delta_eta', 'lepton_delta_phi', 'lepton_delta_r',
                'dilepton_mass', 'dilepton_pt', 'dilepton_eta', 'dilepton_phi', 
                'dilepton_pt_over_m',

                'met_mag', 'met_phi',
                #'n_jets', 'n_fwdjets', 'n_bjets',
                #'bjet_pt', 'bjet_eta', 'bjet_phi', #'bjet_d0',
                #'jet_pt', 'jet_eta', 'jet_phi', #'jet_d0', 
                #'dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi', 
                #'dijet_pt_over_m',

                #'lepton1_b_mass', 'lepton1_b_pt', 
                #'lepton1_b_delta_eta', 'lepton1_b_delta_phi', 'lepton1_b_delta_r',
                #'lepton2_b_mass', 'lepton2_b_pt', 
                #'lepton2_b_delta_eta', 'lepton2_b_delta_phi', 'lepton2_b_delta_r',

                #'dilepton_j_mass', 'dilepton_j_pt', 
                #'dilepton_j_delta_eta', 'dilepton_j_delta_phi', 'dilepton_j_delta_r',
                #'dilepton_b_mass', 'dilepton_b_pt', 
                #'dilepton_b_delta_eta', 'dilepton_b_delta_phi', 'dilepton_b_delta_r',
                #'four_body_mass',
                #'four_body_delta_phi', 'four_body_delta_eta', 'four_body_delta_r',

                #'t_xj', 't_xb', 't_bj'
               ]

    cuts     = 'lepton1_pt > 25 and abs(lepton1_eta) < 2.1 \
                and lepton2_pt > 25 and abs(lepton2_eta) < 2.1 \
                and lepton1_q != lepton2_q and 12 < dilepton_mass < 70' 
                #and n_bjets == 1 and (n_jets > 0 or n_fwdjets > 0)'

    ### Get dataframes with features for each of the datasets ###
    data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                  dataset_names = datasets,
                                  selection     = selection[0],
                                  period        = period,
                                  cuts          = cuts
                                 )

    ### prepare data for training ###
    targets   = ['ttbar', 'zjets', 'bprime_xb']
    dataframe = pd.concat([data_manager.get_dataframe(t) for t in targets])
    dataframe = dataframe[features+targets]
    dataframe = dataframe.fillna(0)
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.iloc[np.random.permutation(dataframe.shape[0])]

    ### scale continuous variables to lie between 0 and 1 ###
    lut = data_manager._lut_features
    for feature in features:
        xmin = lut.loc[feature].xmin
        xmax = lut.loc[feature].xmax
        dataframe[feature] = dataframe[feature].apply(lambda x: (x - xmin)/(xmax - xmin))

    ### vectorize categorical variables ###
    ### split data into training and test data ###
    n_data  = dataframe.shape[0]
    isplit  = int(2*n_data/3)
    train_x = dataframe[features].values[:isplit]
    train_y = dataframe[targets].values[:isplit]
    test_x  = dataframe[features].values[isplit:]
    test_y  = dataframe[targets].values[isplit:]
    
    #df_data = data_manager.get_dataframe('data')
    #df_1b1f = df_data.query(cut_1b1f)
    #df_1b1c = df_data.query(cut_1b1c)
    #df_combined = df_data.query(cut_combined)

    ### model setup ###
    x  = tf.placeholder(tf.float32, [None, len(features)])
    W  = tf.Variable(tf.zeros([len(features), len(targets)]))
    b  = tf.Variable(tf.zeros([len(targets)]))
    y  = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, len(targets)])

    cross_entropy      = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
    train_step         = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ### initialize and run ###
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        acc = []
        n_epochs = 100
        for i in tqdm(range(n_epochs), desc='Training', ncols=75, total=n_epochs):
            sess.run(train_step, feed_dict={x:train_x, y_:train_y})
            acc.append(sess.run(accuracy, feed_dict={x:test_x, y_:test_y}))

        acc = np.array(acc)

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
