#!/usr/bin/env python3

from hyperopt import hp, fmin, tpe

from collections import OrderedDict

from .model import Model
from . import misc

import os

import tensorflow as tf


def preset(nb_layers):
    '''
    Default hyperparameters for the model. Since there are so many of them it is
    more convenient to set them in the source code as opposed to passing
    them as arguments to the CLI. We use an OrderedDict since we want to print the hyperparameters and for that purpose
    keep them in the predefined order.
    '''
    
    return OrderedDict([
        # Hyperparameters for Stochastic Gradient Descent.
        ("learning_rate", 0.005),
        ("momentum", 0.9),
        ("decay", 0.002),
    
        # Number of CNN layers.
        ("nb_layers", 1),
    
        # Hyperparameters for the first convolutional layer.
        ("nb_filters_1", 64),
        ("filter_size_1", 7),
        ("stride_1", (3, 3)),
    
        # Hyperparameter for the first pooling layer.
        ("pool_size_1", (4, 4)),
    
        # Hyperparameters for the second convolutional layer (when two layer
        # architecture is used).
        ("nb_filters_2", 128),
        ("filter_size_2", 3),
        ("stride_2", (2, 2)),
    ])

def hp_space(nb_layers):
    '''
    Returns a hyperopt space of hyperparameters. 
    
    Note:
        For the most part, parameter ranges and distributions were arbitrarily chosen. This should be fixed in the future. As of now, distributions are hard-coded. Soon, we would like to give users the option to input their desired distributions. Ultimately, we would like the distributions themselves to be chosen optimally during runtime.
    '''

    # nb stands for number. E.g, nb_layers is number of layers.
    
    # Initialize search space.
    space = OrderedDict()
    
    # Hyperparameters for Stochastic Gradient Descent.
    space['learning_rate'] = hp.uniform('learning_rate', 0.001, 1)
    space['momentum'] = hp.uniform('momentum', 0.1, 1)
    space['decay'] = hp.uniform('decay', 0.001, 1)
    
    # Add number of CNN layers to the hyperparameters dict.
    space['nb_layers'] = hp.choice('nb_layers', [nb_layers])
    
    # Hyperparameters for each CNN layer.
    for i in range(1,nb_layers+1):
        space['nb_filters_'+str(i)] = hp.choice('nb_filters_'+str(i), [i for i in range(1, 128)])
        space['filter_size_'+str(i)] = hp.choice('filter_size_'+str(i), [i for i in range(1, 10)])
        space['stride_'+str(i)] = hp.choice('stride_'+str(i), [(3,3)])
        
        if i < nb_layers or nb_layers == 1:
            space['pool_size_'+str(i)] = hp.choice('pool_size_'+str(i), [(4,4)])
    
    return space


def find_best(source_model, features, labels, nb_layers = 1, max_evals = 100, epochs_per_eval = 10):
    '''
    Run a hyperparameter sweep to find the best hyperparameters, defined as ones that have the lowest loss in a small test run. In the future, we would like to have the hyperparameters be optimized during training.
    
    Inputs:
        source_model - Model object - Required - The parent Model from which to get various values, such as dataset and num_channels.
        features, labels - ? - Required - The training set on which to run the training epochs.
        nb_layers - int - Optional - number of CNN layers. Default is 1.
        max_evals - int - Optional - Maximum number of sets of hyperparameters to test. Default is 100.
        epochs_per_eval - int - Optional - The number of epochs per evaluation to conduct. Default is 10. The total number of epochs in the hypersweep is max_evals*epochs_per_eval. 
        
    Returns:
        A dict of hyperparameters.
    '''
       
    return fmin(
        fn = lambda hyperparameters: objective(source_model, hyperparameters, features, labels, epochs = epochs_per_eval), 
        space = hp_space(nb_layers), 
        algo = tpe.suggest, 
        max_evals = max_evals
    )


def objective(source_model, hyperparameters, features, labels, epochs = 10):
    '''
    Outputs the objective function value for a specific model. Used in find_best.
    
    The objective function to be minimized is the model's loss on test data.
    
    Options:
        source_model - Model object - Required - The parent Model from which to get various values, such as dataset and num_channels.
        features, labels - ? - Required - The training set on which to run the training epochs.
        epochs - int - Optional - The number of training epochs.
    '''
    
    print('Training with the following hyperparameters: ')
    print(hyperparameters)
    misc.print_line()
    #global hps_counter
    
    # First, initialize the model, getting values from source_model.
    model = source_model.spawn_child(excluded_attributes = {'model', 'hyperparameters'}, subdir = 'hp_sweep')
    
    with tf.Session() as session:
        # Initialize the model with the hyperparameters.
        model.init(
            hyperparameters = hyperparameters
        )

        # Next, train the model on the set number of epochs, and save the history.
        history = model.train(
            features, 
            labels, 
            epochs = epochs
        )

        # Finally, use the loss value of the final epoch. This might use some fine-tuning, perhaps instead using the average loss or the minimum loss.
        loss = history.history['loss'][-1]

        # Remove the model from memory to save space.
        model_dir = model.model_dir
        model.unload_model()
        #del model
        print("The model at '" + model_dir + " was unloaded from memory.")
        misc.print_line()
    
    return loss