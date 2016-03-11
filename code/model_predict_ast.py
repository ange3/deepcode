#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: model_predict_ast.py
# @Author: Lisa Wang
# @created: Feb 18 2016
#
#==============================================================================
# DESCRIPTION:
# model functions for predicting the ast in a trajectory using RNNs.
#==============================================================================
# CURRENT STATUS: In progress/ working! :) 
#==============================================================================
# USAGE: 
# import model_predict_ast
#==============================================================================
#
###############################################################################

import lasagne
import numpy as np
import theano
import theano.tensor as T
import time
import utils

def _build_net_layers(num_timesteps, num_asts, hidden_size, learning_rate, grad_clip=10, dropout_p=0.0, num_lstm_layers=1):
    print("Building network ...")
   
    # First, we build the network, starting with an input layer
    l_in = lasagne.layers.InputLayer(shape=(None, num_timesteps, num_asts))

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

    for i in xrange(num_lstm_layers):
        if i == 0:
            l_lstm = lasagne.layers.LSTMLayer(
                l_in, hidden_size, grad_clipping=grad_clip,
                nonlinearity=lasagne.nonlinearities.tanh)
        else:
            l_lstm = lasagne.layers.LSTMLayer(
                l_lstm, hidden_size, grad_clipping=grad_clip,
                nonlinearity=lasagne.nonlinearities.tanh)


    # The l_forward layer creates an output of dimension (batch_size, num_timesteps, hidden_size)


    l_reshape = lasagne.layers.ReshapeLayer(l_lstm, (-1, hidden_size))
    l_dropout = lasagne.layers.DropoutLayer(l_reshape, p=dropout_p, rescale=True)
    l_out = lasagne.layers.DenseLayer(l_dropout,
        num_units=num_asts,
        W = lasagne.init.Normal(),
        nonlinearity=lasagne.nonlinearities.softmax)
    # just reshaping, so we have the num_timesteps dimension back.
    l_out = lasagne.layers.ReshapeLayer(l_out, (-1, num_timesteps, num_asts))
    # after l_out, we have shape (batchsize, num_timesteps, num_asts)
    # and the values should be probabilities for the asts
    # for each trajectory, at each time step, we compute probabilities
    # over all asts. The probabilities should sum up to 1, that's why we use
    # softmax nonlinearity.
    l_out_slice = lasagne.layers.SliceLayer(l_out, indices=-1, axis=1)
    
    return l_in, l_out, l_out_slice


def create_model(num_timesteps, num_asts, hidden_size, learning_rate, grad_clip=10, dropout_p=0.5, num_lstm_layers=1):
    '''
     returns train function which reports both loss and accuracy
     and test function, which also reports both loss and accuracy
    '''
    
    l_in, l_out, l_out_slice = _build_net_layers(num_timesteps, num_asts, hidden_size, learning_rate, grad_clip, dropout_p, num_lstm_layers)
    # pred should be of shape (batchsize, num_timesteps, num_asts)
    pred = lasagne.layers.get_output(l_out)
    # pred_slice should be of shape (batchsize, num_asts), only contains
    # predictions for the last timestep
    pred_slice = lasagne.layers.get_output(l_out_slice)
    # truth should also be of shape (batchsize, num_timesteps, num_asts)
    truth = T.imatrix("truth")
    pred_2d = pred.reshape((-1, num_asts))
    truth_1d = truth.reshape((-1,))

    # pred_2d_shape = T.shape(pred_2d)
    # truth_1d_shape = T.shape(truth_1d)

    # categorical_crossentropy
    loss = T.nnet.categorical_crossentropy(pred_2d, truth_1d).mean()
    # categorical accuracy
    # acc = T.nnet.categorical_crossentropy(pred_2d, truth_1d).mean()
    acc = lasagne.objectives.categorical_accuracy(pred_2d, truth_1d).mean()
    # update function
    print("Computing updates ...")
    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adam(loss, all_params, learning_rate)

    # training function
    print("Compiling functions ...")
    train_loss = theano.function([l_in.input_var, truth], loss, updates=updates, allow_input_downcast=True)
    compute_loss = theano.function([l_in.input_var, truth], loss, allow_input_downcast=True)
    # training function, returns loss and acc
    train_loss_acc = theano.function([l_in.input_var, truth], [loss, acc, pred], updates=updates, allow_input_downcast=True)
    # computes loss and accuracy, without training
    compute_loss_acc = theano.function([l_in.input_var, truth], [loss, acc, pred], allow_input_downcast=True)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs. 
    probs = theano.function([l_in.input_var], pred_slice, allow_input_downcast=True)

    print("Compiling done!")
    
    return train_loss_acc, compute_loss_acc, probs # , pred_2d_shape, truth_1d_shape

    
def _compute_loss_acc_pred(X, truth, compute_loss_acc):
    # a full pass over the given data without training
    # returns loss, raw accuracy, corrected accuracy and predictions
    loss, acc, pred = compute_loss_acc(X, truth)
    corrected_acc = compute_corrected_acc_on_ast_rows(X, truth, pred)
    return loss, acc, corrected_acc, pred


def train(train_data, val_data, train_loss_acc, compute_loss_acc, num_epochs=5, batchsize=32, record_per_iter=False):
    X_train, truth_train = train_data
    X_val, truth_val = val_data

    num_train = X_train.shape[0]
    num_val = X_val.shape[0]
    total_train_iters = (num_train / batchsize) * num_epochs
    num_iters = 0


    print("Starting training...")
    print("Total training iterations: {}".format(total_train_iters))
    # We iterate over epochs:
    train_accs = []
    val_accs = []
    train_corrected_accs = []
    val_corrected_accs = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_loss_ep = 0.0
        train_acc_ep = 0.0
        train_corrected_acc_ep = 0.0
        train_batches = 0
        start_time = time.time()
        for batch in utils.iter_minibatches([X_train, truth_train], batchsize, shuffle=False):
            X_, truth_ = batch
            train_loss, train_acc, train_pred = train_loss_acc(X_, truth_)
            train_corrected_acc = compute_corrected_acc_on_ast_rows(X_, truth_, train_pred)
            # train_loss_ep += train_loss
            # train_acc_ep += train_acc
            # train_corrected_acc_ep += train_corrected_acc
            # train_batches += 1
            num_iters += 1
            val_loss, val_acc, val_corrected_acc, pred_val = _compute_loss_acc_pred(X_val, truth_val, compute_loss_acc)
            print("Ep {} \titer {}  \tloss {:.5f}, train acc {:.2f}, train corr acc {:.2f}, val acc {:.2f}, val corr acc {:.2f}".format(epoch, num_iters, float(train_loss), train_acc * 100, train_corrected_acc * 100, val_acc *100, val_corrected_acc *100) )
        
            if record_per_iter:
                train_accs.append(train_acc)
                train_corrected_accs.append(train_corrected_acc)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                val_corrected_accs.append(val_corrected_acc)

        # train_acc = train_acc_ep/train_batches
        # train_corrected_acc = train_corrected_acc_ep/train_batches
        # train_loss = train_loss_ep/train_batches
        
        # train_loss, train_acc, train_corrected_acc, pred_train = _compute_loss_acc_pred(X_train[:num_val,:,:], truth_train[:num_val,:], compute_loss_acc)
        train_loss, train_acc, train_corrected_acc, pred_train = _compute_loss_acc_pred(X_train, truth_train, compute_loss_acc)        
        if not record_per_iter:
            # recording values for each epoch
            train_accs.append(train_acc)
            train_corrected_accs.append(train_corrected_acc)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_corrected_accs.append(val_corrected_acc)

        # Then we print the results for this epoch:
        print("\nEpoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(float(train_loss)))
        print("  training raw accuracy:\t\t{:.2f} %".format(train_acc * 100))
        print("  training corrected acc:\t\t{:.2f} %".format(train_corrected_acc * 100))

        print("  validation loss:\t\t{:.6f}".format(float(val_loss)))
        print("  validation raw accuracy:\t\t{:.2f} %".format(val_acc * 100))
        print("  validation corrected acc:\t\t{:.2f} % \n".format(val_corrected_acc * 100))

    print("Training completed.")
    return train_losses, train_accs, train_corrected_accs, val_losses, val_accs, val_corrected_accs



def compute_corrected_acc_on_ast_rows(X, truth, pred):
    batchsize, num_timesteps, num_asts = X.shape
    correct_count = 0
    total_count = 0
    for n in xrange(batchsize):
        for t in xrange(num_timesteps):
            if np.argmax(X[n,t,:]) != 0:
                # only include predictions not on the <END> token
                total_count += 1
                if np.argmax(pred[n,t,:]) == truth[n,t]:
                    correct_count += 1
    corrected_acc = correct_count/float(total_count)
    return corrected_acc


def compute_corrected_acc_on_ast_ids(X_ast_ids, truth_ast_ids, pred_ast_ids): 
    batchsize, num_timesteps = X_ast_ids.shape
    correct_count = 0
    total_count = 0
    for n in xrange(batchsize):
        for t in xrange(num_timesteps):
            if X_ast_ids[n,t] != -1:
                # only include predictions not on the <END> token
                total_count += 1
                if pred_ast_ids[n,t] == truth_ast_ids[n,t]:
                    correct_count += 1
    corrected_acc = correct_count/float(total_count)
    return corrected_acc



def check_accuracy(data, compute_loss_acc, dataset_name):
    X, truth = data
    print("Testing...")
    # After training, we compute and print the test error:

    loss, raw_acc, corrected_acc, pred = _compute_loss_acc_pred(X, truth, compute_loss_acc)

    print("Final results:")
    print("  {} loss:\t\t\t{:.6f}".format(dataset_name, loss * 1.0))
    print("  {} raw accuracy:\t\t{:.2f} %".format(dataset_name, raw_acc * 100))
    print("  {} corrected accuracy:\t{:.2f} %".format(dataset_name, corrected_acc * 100))

    return loss, raw_acc, corrected_acc, pred

