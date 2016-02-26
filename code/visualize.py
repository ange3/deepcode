#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: visualize.py
# @Author: Lisa Wang
# @created: Jan 30 2016
#==============================================================================
# DESCRIPTION:
# For plotting and visulizing our data and results.
#==============================================================================
# CURRENT STATUS: In progress/ working! :) 
#==============================================================================
# USAGE:
# import visualize
#==============================================================================
#
###############################################################################

import matplotlib.pyplot as plt
import time
import numpy as np

def smoothen_data(data, smooth_window=100):
    smooth = []
    for i in xrange(len(data)-smooth_window):
        smooth.append(np.mean(data[i:i+smooth_window]))

    for i in xrange(len(data)-smooth_window, len(data)):
        smooth.append(np.mean(data[i:len(data)]))
    return smooth

def plot_loss_acc(data_set, losses, train_accs, val_accs, lr, rg, ep, num_train,  xlabel='epochs'):              
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    train_accs_line, = ax1.plot(xrange(len(train_accs)), train_accs, 'b-', label='train accuracies')
    val_accs_line, = ax1.plot(xrange(len(val_accs)), val_accs, 'g-', label='val accuracies')

    ax1.set_ylabel('accuracies', color='b')
    ax1.set_xlabel(xlabel)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    losses_line, = ax2.plot(xrange(len(losses)), losses, 'r-', label='losses')
    ax2.set_ylabel('losses', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.legend(handles=[losses_line, train_accs_line, val_accs_line],bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand")

    plt.show()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    figure_filename = "../loss_plots/{}_lr{}_rg{}_ep{}_num_train{}_{}.png".format(data_set, lr, rg, ep, num_train, timestr)
    fig.savefig(figure_filename)

def plot_loss_train_test_acc(data_set, losses, train_accuracies, test_accuracies, lr, rg, ep, num_train):
    smooth_accs_train = smoothen_data(train_accuracies)
    smooth_accs_test = smoothen_data(test_accuracies)
                        
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    train_accs_line, = ax1.plot(xrange(len(smooth_accs_train)), smooth_accs_train, 'b-', label='train accuracies')
    test_accs_line, = ax1.plot(xrange(len(smooth_accs_test)), smooth_accs_test, 'g-', label='test accuracies')
    ax1.set_ylabel('accuracies', color='b')
    ax1.set_xlabel('iterations')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    losses_line, = ax2.plot(xrange(len(losses)), losses, 'r-', label='losses')
    ax2.set_ylabel('losses', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.legend(handles=[losses_line, train_accs_line, test_accs_line],bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand")
    # plt.legend([losses_line, train_accs_line, test_accs_line], ['losses', 'train accuracies', 'test accuracies'])
    plt.show()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    figure_filename = "../loss_plots/{}_lr{}_rg{}_ep{}_n{}_{}.png".format(data_set, lr, rg, ep, num_train, timestr)
    fig.savefig(figure_filename)