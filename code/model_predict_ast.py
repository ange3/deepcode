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

    
def _check_val_loss_acc(X_val, truth_val, batchsize, compute_loss_acc):
    # a full pass over the validation data:
    val_err = 0.0
    val_acc = 0.0
    val_batches = 0
    for batch in utils.iter_minibatches([X_val, truth_val], batchsize, shuffle=False):
        X_, truth_ = batch
        err, acc, pred = compute_loss_acc(X_, truth_)
        val_err += err
        val_acc += acc
        val_batches += 1
    val_loss = val_err/val_batches
    val_acc = val_acc/val_batches * 100
    return val_loss, val_acc


def train(train_data, val_data, train_loss_acc, compute_loss_acc, num_epochs=5, batchsize=32):
    
    X_train, truth_train = train_data
    X_val, truth_val = val_data
    print("Starting training...")
    # We iterate over epochs:
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0.0
        train_acc = 0.0
        train_batches = 0
        start_time = time.time()
        for batch in utils.iter_minibatches([X_train, truth_train], batchsize, shuffle=False):
            X_, truth_ = batch
            err, acc, pred = train_loss_acc(X_, truth_)
            train_err += err
            train_acc += acc
            train_batches += 1
            val_loss, val_acc = _check_val_loss_acc(X_val, truth_val, batchsize, compute_loss_acc)
            print("  Epoch {} \tbatch {} \tloss {} \ttrain acc {:.2f} \tval acc {:.2f} ".format(epoch, train_batches, err, acc * 100, val_acc) )
        train_acc = train_acc/train_batches * 100
        train_accuracies.append(train_acc)
        train_loss = train_err/train_batches
        train_losses.append(train_loss)

        val_loss, val_acc = _check_val_loss_acc(X_val, truth_val, batchsize, compute_loss_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_loss))
        print("  training accuracy:\t\t{:.2f} %".format(train_acc))
        print("  validation loss:\t\t{:.6f}".format(val_loss))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc))

    print("Training completed.")
    return train_losses, train_accuracies, val_accuracies

   
def get_predicted_ast_ids(pred, truth, row_to_ast_id_map):
    '''
    Takes in a prediction matrix and returns the predicted AST ID at each sample and each timestep 

    pred: (batchsize, num_timesteps, num_asts)
    truth: (batchsize, num_timesteps)

    pred_output: (batchsize, num_timesteps)
    '''

    batchsize, num_timesteps, num_asts = pred.shape
    pred_output = np.zeros((batchsize, num_timesteps))
    truth_output = np.zeros((batchsize, num_timesteps))

    for n in xrange(batchsize):
        for t in xrange(num_timesteps):
            pred_output[n,t] = row_to_ast_id_map[np.argmax(pred[n,t,:])]
            truth_output[n,t] = row_to_ast_id_map[int(truth[n,t])]

    return pred_output, truth_output


def check_accuracy(data, compute_loss_acc, row_to_ast_id_map, dataset_name='test', batchsize=32):
    X_test, truth_test = data
    print("Testing...")
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in utils.iter_minibatches([X_test, truth_test], batchsize, shuffle=False):
        X_, truth_ = batch
        err, acc, pred = compute_loss_acc(X_, truth_)
        test_err += err
        test_acc += acc
        test_batches += 1
        predicted_ast_ids, truth_ast_ids = get_predicted_ast_ids(pred, truth_, row_to_ast_id_map)
        print("Predicted AST IDs")
        print predicted_ast_ids[:10,:]
        print ("Truth AST IDs")
        print truth_ast_ids[:10, :]

    print("Final results:")
    print("  {} loss:\t\t\t{:.6f}".format(dataset_name, test_err / test_batches))
    print("  {} accuracy:\t\t{:.2f} %".format(dataset_name, test_acc / test_batches * 100))



