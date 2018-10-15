#!/usr/bin/env python3

import argparse
import time
import os
import sys

import waterNet.hyperparameters as hp

def create_parser():
    parser = argparse.ArgumentParser(description="Train a convolutional neural network to predict water in satellite images.")

    parser.add_argument(
        "-p, --preprocess-data",
        dest="preprocess_data",
        action="store_const",
        const=True,
        default=False,
        help="When selected preprocess data.")
    parser.add_argument(
        "-i, --init-model",
        dest="init_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected initialise model.")
    parser.add_argument(
        "-t, --train-model",
        dest="train_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected train model.")
    parser.add_argument(
        "-e, --evaluate-model",
        dest="evaluate_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected evaluatel model.")
    parser.add_argument(
        "-d, --debug",
        dest="debug",
        action="store_const",
        const=True,
        default=False,
        help="Run on a small test dataset.")
 #   parser.add_argument(
 #       "-a, --architecture",
 #       dest="architecture",
 #       default="one_layer",
 #       choices=["one_layer", "two_layer"],
 #       help="Neural net architecture.")
    
    parser.add_argument(
        "-v, --visualise",
        dest="visualise",
        default=False,
        action="store_const",
        const=True,
        help="Visualise labels.")
    parser.add_argument(
        "-T, --tensorboard",
        dest="tensorboard",
        default=False,
        action="store_const",
        const=True,
        help="Store tensorboard data while training.")
    parser.add_argument(
        "-C, --checkpoints",
        dest="checkpoints",
        default=False,
        action="store_const",
        const=True,
        help="Create checkpoints while training.")
    parser.add_argument(
        "--dataset",
        default="sentinel",
        choices=["sentinel"],
        help="Determine which dataset to use.")
    
    parser.add_argument(
        "--nb-layers",
        dest="nb_layers",
        type=int,
        default=1,
        help="The number of layers to use in the neural network. Default is 1."
        )
        
    parser.add_argument(
        "--tile-size", default=64, type=int, help="Choose the tile size.")
    parser.add_argument(
        "--num-channels", default=3, type=int, help="Set the number of channels in the dataset.")
    parser.add_argument(
        "--epochs", default=10, type=int, help="Number of training epochs.")
    parser.add_argument(
        "--model-id",
        default=None,
        type=str,
        help="Model that should be used. Must be an already existing ID.")
    parser.add_argument(
        "--setup",
        default=False,
        action="store_const",
        const=True,
        help="Create all necessary directories for the classifier to work.")
    parser.add_argument(
        "--out-format",
        default="GeoTIFF",
        choices=["GeoTIFF", "Shapefile"],
        help="Determine the format of the output for the evaluation method.")
    
    parser.add_argument(
        "--data-dir",
        default="data",
        type=str,
        help="Set the data directory, relative to where waterNet.py is being run. Default is 'data'.")
    
    parser.add_argument(
        "--hp-sweep",
        default=False,
        #type=bool,
        action="store_const",
        const=True,
        help="WaterNet defaults to using the preset hyperparameters (defined in waterNet/config.py). Use the '--hp-sweep' argument to do a hyperparameter sweep (run small training runs to determine the best hyperparameters)."
    )
    
    parser.add_argument(
        "--hp-sweep-evals",
        dest="hp_sweep_evals",
        type=int,
        default=100,
        help="The number of sets of hyperparameters to test during the sweep. Default is 100."
        )
    
    parser.add_argument(
        "--hp-sweep-epochs",
        dest="hp_sweep_epochs",
        type=int,
        default=100,
        help="The number of epochs to train per set of hyperparameters. Default is 100."
        )

    return parser


def main():
    
    parser = create_parser()
    args = parser.parse_args()
    
    os.environ['DATA_DIR'] = args.data_dir
    
    from waterNet.config import DATASETS, OUTPUT_DIR, TRAIN_DATA_DIR, LABELS_DIR, MODELS_DIR, HYPERPARAMETERS
    from waterNet.preprocessing import preprocess_data
    from waterNet.model import init_model, train_model, compile_model, Model
    from waterNet.evaluation import evaluate_model
    from waterNet.io_util import save_makedirs, save_model_summary, load_model, create_directories
    from waterNet.geo_util import visualise_labels

  
    if args.setup:
        create_directories()

    if args.debug:
        dataset = DATASETS["debug"]
        args.dataset = "debug"
        features, _, labels, _ = preprocess_data(
            args.tile_size, dataset=dataset)
        features_train, features_test = features[:100], features[100:120]
        labels_train, labels_test = labels[:100], labels[100:120]
    
    elif args.train_model or args.evaluate_model or args.preprocess_data:
        dataset = DATASETS[args.dataset]
        load_from_cache = not args.preprocess_data
        try:
            features_train, features_test, labels_train, labels_test = preprocess_data(
                args.tile_size, dataset=dataset, only_cache=load_from_cache)
        except IOError:
            print("Cache file does not exist. Please run again with -p flag.")
            sys.exit(1)

        if args.visualise:
            visualise_labels(labels_train, args.tile_size, LABELS_DIR)
            visualise_labels(labels_test, args.tile_size, LABELS_DIR) 
    
    if not args.model_id:
        timestamp = time.strftime("%d_%m_%Y_%H%M%S")
        model_id = "{}_{}_{}".format(timestamp, args.dataset, str(args.nb_layers)+'L')
        only_load_model = False
    else:
        model_id = args.model_id
        only_load_model = True # only_load_model is used to stop Model() from creating the model if it did not exist.
    
    
    if args.init_model or args.train_model or args.evaluate_model:
        model = Model(
            model_id = model_id, 
            models_dir = MODELS_DIR,
            only_load = only_load_model,
            dataset = dataset,
            tile_size = args.tile_size,
            num_channels = args.num_channels
        ) 
        
    if args.hp_sweep:
        
        print('Running a hyperparameter sweep for ' + str(args.hp_sweep_evals) + ' sets of hyperparameters, with ' + str(args.hp_sweep_epochs) + ' training epochs per set.')
        
        hyperparameters = hp.find_best(
            source_model = model,
            features = features_train, 
            labels = labels_train, 
            nb_layers = args.nb_layers, 
            max_evals = args.hp_sweep_evals, 
            epochs_per_eval = args.hp_sweep_epochs,
        )
        
        print('Successfully completed the hyperparameter sweep. The best performing hyperparameters are: ')
        print(hyperparameters)

    else:
        hyperparameters = HYPERPARAMETERS
        hyperparameters['nb_layers'] = args.nb_layers
       

    if args.init_model:
        model.init(
            hyperparameters = hyperparameters, 
        )
        model.summary()
        
    elif args.train_model or args.evaluate_model:
        model.compile()

    if args.train_model:
        model.train(
            features_train, 
            labels_train, 
            epochs = args.epochs, 
            checkpoints = args.checkpoints, 
            tensorboard = args.tensorboard
        )

    if args.evaluate_model:
        model.evaluate(
            features_test, 
            labels_test, 
            out_format = args.out_format)
        
    # Save the Model object.
    model.save()


if __name__ == '__main__':
    main()