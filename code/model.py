#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: model.py
# @Author: Lisa Wang
# @created: Feb 18 2016
#
#==============================================================================
# DESCRIPTION:
# 
#==============================================================================
# CURRENT STATUS: In progress/ working! :) 
#==============================================================================
# USAGE: 
# import model
#==============================================================================
#
###############################################################################

import lasagne
import numpy as np
import theano
import theano.tensor as T
import time
import utils

def _build_net_layers_dropout(num_timesteps, num_problems, hidden_size, learning_rate, grad_clip=10, dropout_p=0.0, num_lstm_layers=1):
    """
    Input layer
    [LSTM layer] * num_lstm_layers
    Reshape layer
    Dropout layer
    Dense layer
    Output layer
    
    """
    l_in = lasagne.layers.InputLayer(shape=(None, num_timesteps, num_problems * 2))
    for i in xrange(num_lstm_layers):
        if i == 0:
            l_lstm = lasagne.layers.LSTMLayer(
                l_in, hidden_size, grad_clipping=grad_clip,
                nonlinearity=lasagne.nonlinearities.tanh)
        else:
            l_lstm = lasagne.layers.LSTMLayer(
                l_lstm, hidden_size, grad_clipping=grad_clip,
                nonlinearity=lasagne.nonlinearities.tanh)
    l_reshape = lasagne.layers.ReshapeLayer(l_lstm, (-1, hidden_size))
    l_dropout = lasagne.layers.DropoutLayer(l_reshape, p=dropout_p, rescale=True)
    l_out = lasagne.layers.DenseLayer(l_dropout,
        num_units=num_problems,
        W = lasagne.init.Normal(),
        nonlinearity=lasagne.nonlinearities.sigmoid)
    l_out = lasagne.layers.ReshapeLayer(l_out, (-1, num_timesteps, num_problems))

    return l_in, l_out
    

def create_model(num_timesteps, num_problems, hidden_size, learning_rate, grad_clip=10, dropout_p=0.5, num_lstm_layers=1):
    '''
     returns train function which reports both loss and accuracy
     and test function, which also reports both loss and accuracy
    '''
    
    l_in, l_out = _build_net_layers_dropout(num_timesteps, num_problems, hidden_size, learning_rate, grad_clip, dropout_p, num_lstm_layers)
    ''' pred:for each student, a vector that gives probability of next question being answered correctly
        y: for each student, a one-hot-encoding of shape (num_timesteps, num_problems), indicating which problem the student will do at the next timestep 
        truth: for each student, a vector that indicates for each timestep whether next problem was answered correctly
    '''
    pred = lasagne.layers.get_output(l_out)
    next_problem = T.tensor3('next_problem')
    truth = T.imatrix("truth")


    # pred_probs: shape(num_samples, num_timesteps)
    # we reduce the three-dimensional probability tensor to two dimensions
    # by only keeping the probabilities corresponding to the next problem
    # we don't care about the predicted probabilities for other problems
    pred_probs = T.sum(pred * next_problem, axis = 2)
    # loss function
    loss = T.nnet.binary_crossentropy(pred_probs, truth)

    # TODO: why do we use both cost and loss? (naming :D)
    # take average of loss per sample, makes loss more comparable when batch sizes change
    cost = loss.mean()

    # update function
    print("Computing updates ...")
    all_params = lasagne.layers.get_all_params(l_out)
    updates = lasagne.updates.adam(cost, all_params, learning_rate)

    # Function to compute accuracy:
    # if probability was > 0.5, we consider it a 1, and 0 otherwise.
    # binary_pred: for each student, a vector of length num_timesteps indicating whether the 
    # probability of getting the next problem correct is larger than 0.5.
    binary_pred = T.round(pred_probs)
    acc = T.mean(T.eq(binary_pred, truth))

    # training function
    print("Compiling functions ...")
    # training function, only returns loss
    train_fn = theano.function([l_in.input_var, next_problem, truth], cost, updates=updates, allow_input_downcast=True)
    # training function, returns loss and acc
    train_acc_fn = theano.function([l_in.input_var, next_problem, truth], [cost, acc], updates=updates, allow_input_downcast=True)
    # computes loss
    compute_loss = theano.function([l_in.input_var, next_problem, truth], cost, allow_input_downcast=True)
    # computes loss and accuracy
    compute_loss_acc = theano.function([l_in.input_var, next_problem, truth], [cost, acc], allow_input_downcast=True)

    print("Compiling done!")
    
    return train_acc_fn, compute_loss_acc


def _check_val_loss_acc(X_val, next_problem_val, truth_val, batchsize, compute_cost_acc):
    # a full pass over the validation data:
    val_err = 0.0
    val_acc = 0.0
    val_batches = 0
    for batch in utils.iterate_minibatches(X_val, next_problem_val, truth_val, batchsize, shuffle=False):
        X_, next_problem_, truth_ = batch
        err, acc = compute_cost_acc(X_, next_problem_, truth_)
        val_err += err
        val_acc += acc
        val_batches += 1
    val_loss = val_err/val_batches
    val_acc = val_acc/val_batches * 100
    return val_loss, val_acc


def train(train_data, val_data, train_acc_fn, compute_cost_acc, num_epochs=5, batchsize=32):
    
    X_train, next_problem_train, truth_train = train_data
    X_val, next_problem_val, truth_val = val_data
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
        for batch in utils.iterate_minibatches(X_train, next_problem_train, truth_train, batchsize, shuffle=False):
            X_, next_problem_, truth_ = batch
            err, acc = train_acc_fn(X_, next_problem_, truth_)
            train_err += err
            train_acc += acc
            train_batches += 1
            val_loss, val_acc = _check_val_loss_acc(X_val, next_problem_val, truth_val, batchsize, compute_cost_acc)
            print("  Epoch {} \tbatch {} \tloss {} \ttrain acc {:.2f} \tval acc {:.2f} ".format(epoch, train_batches, err, acc * 100, val_acc) )
        train_acc = train_acc/train_batches * 100
        train_accuracies.append(train_acc)
        train_loss = train_err/train_batches
        train_losses.append(train_loss)

        val_loss, val_acc = _check_val_loss_acc(X_val, next_problem_val, truth_val, batchsize, compute_cost_acc)
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

    

def check_accuracy(data, compute_cost_acc, dataset_name='test', batchsize=32):
    X_test, next_problem_test, truth_test = data
    print("Testing...")
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in utils.iterate_minibatches(X_test, next_problem_test, truth_test, batchsize, shuffle=False):
        X_, next_problem_, truth_ = batch
        err, acc = compute_cost_acc(X_, next_problem_, truth_)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  {} loss:\t\t\t{:.6f}".format(dataset_name, test_err / test_batches))
    print("  {} accuracy:\t\t{:.2f} %".format(dataset_name, test_acc / test_batches * 100))


