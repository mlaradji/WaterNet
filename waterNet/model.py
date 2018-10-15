"""Implementation of the convolutional neural net."""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import numpy as np

from . import io_util, misc
from .config import MODELS_DIR, HYPERPARAMETERS

import pickle
import copy
import time

def train_model(
    model,
    features,
    labels,
    tile_size,
    model_id,
    epochs=10,
    checkpoints=False,
    tensorboard=False
):
    """Train a model with the given features and labels."""

    # The features and labels are a list of triples when passed
    # to the function. Each triple contains the tile and information
    # about its source image and its postion in the source. To train
    # the model we extract just the tiles.
    X, y = get_matrix_form(features, labels, tile_size)

    X = normalise_input(X)

    # Directory which is used to store the model and its weights.
    model_dir = os.path.join(MODELS_DIR, model_id)

    checkpointer = None
    if checkpoints:
        checkpoints_file = os.path.join(model_dir, "weights.hdf5")
        checkpointer = ModelCheckpoint(checkpoints_file)

    tensorboarder = None
    if tensorboard:
        log_dir = os.path.join(TENSORBOARD_DIR, model_id)
        tensorboarder = TensorBoard(log_dir=log_dir)

    callbacks = [c for c in [checkpointer, tensorboarder] if c]

    print("Start training.")
    history = model.fit(X, y, epochs=epochs, callbacks=callbacks, validation_split=0.1)

    #io_util.save_model(model, model_dir)
    return model, history


def init_model(
    #model_id,
    tile_size,
    num_channels,
    hyperparameters = HYPERPARAMETERS
              ):
    """Initialise a new model with the given hyperparameters and save it for later use."""

    model = Sequential()
    
    hp = hyperparameters

    for i in range(1,hp['nb_layers']+1):
    
        model.add(
            Conv2D(
                filters = hp['nb_filters_'+str(i)],
                kernel_size = (hp['filter_size_'+str(i)],hp['filter_size_'+str(i)]), #***FIXME***
                strides = hp['stride_'+str(i)],
                input_shape = (tile_size, tile_size, num_channels))
        )
        
        model.add(Activation('relu'))
        
        if i < hp['nb_layers'] or hp['nb_layers'] == 1:
            model.add(MaxPooling2D(pool_size = hp['pool_size_'+str(i)]))
    
    model.add(Flatten())
    model.add(Dense(tile_size * tile_size))
    model.add(Activation('sigmoid'))  
    
    

    model = compile_model(model, hp['learning_rate'], hp['momentum'], hp['decay'])

    # Print a summary of the model to the console.
    model.summary()

    #model_dir = os.path.join(MODELS_DIR, model_id)
    #io_util.save_makedirs(model_dir)
    
    #io_util.save_model(model, model_dir)

    return model

'''
    if architecture == 'one_layer':
        model.add(
            Convolution2D(
                nb_filters_1,
                filter_size_1,
                filter_size_1,
                subsample=stride_1,
                input_shape=(tile_size, tile_size, num_channels)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_1))
        model.add(Flatten())
        model.add(Dense(tile_size * tile_size))
        model.add(Activation('sigmoid'))
    elif architecture == 'two_layer':
        model.add(
            Convolution2D(
                nb_filters_1,
                filter_size_1,
                filter_size_1,
                subsample=stride_1,
                input_shape=(tile_size, tile_size, num_channels)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_1))
        model.add(
            Convolution2D(
                nb_filters_2,
                filter_size_2,
                filter_size_2,
                subsample=stride_2))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(tile_size * tile_size))
        model.add(Activation('sigmoid'))

'''


def compile_model(model, learning_rate, momentum, decay):
    """Compile the keras model with the given hyperparameters."""

    optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    return model

def normalise_input(features):
    """Normalise the features such that all values are in the range [0,1]."""
    features = features.astype(np.float32)

    return np.multiply(features, 1.0 / 255.0)


def get_matrix_form(features, labels, tile_size):
    """Transform a list of triples of features and labels. To a matrix which contains
    only the tiles used for training the model."""
    features = [tile for tile, position, path in features]
    labels = [tile for tile, position, path in labels]

    # The model will have one output corresponding to each pixel in the feature tile.
    # So we need to transform the labels which are given as a 2D bitmap into a vector.
    labels = np.reshape(labels, (len(labels), tile_size * tile_size))
    return np.array(features), np.array(labels)



##########
##########


class Model(object):
    '''
    ***FIXME***
    '''
    
    def __init__(self, model_id = None, models_dir = MODELS_DIR, only_load = False, hyperparameters = None, dataset = None, tile_size = None, num_channels = None):
        '''
        Set the model id and the directory to store the model. If the model id already exists, that model will be loaded.
        
        Inputs:
            model_id - str - Optional - Defaults to "%d_%m_%Y_%H%M". - If set, it will attempt to load a model with the same model_id.
            output_dir - str - Optional - Defaults to $OUTPUT_DIR defined in .config.py, which default value is '$data_dir/working/models'.
            only_load - bool - Optional - Set to True to skip creating the model if it doesn't exist. Default is False.
            hyperparameters - dict - Optional - Defaults to None.
            dataset - str - Optional - Set the name of the dataset. - Defaults to None.
            tile_size - tuple - Optional - Tile size to use for the dataset. - Defaults to None.
            num_channels - int - Optional - The number of channels in the dataset. - Defaults to None.
        '''

        
        if model_id is None:
            timestamp = time.strftime("%d_%m_%Y_%H%M%S")
            model_id = "{}".format(timestamp)
        
        self.model_id = model_id
        
        self.model_dir = os.path.join(models_dir, self.model_id)
        
        # Load the model if it exists.
        try:
            self.load()
            print("The model '"+self.model_dir+"' has been loaded.")
            
        # If it doesn't exist, create the model directory.
        except FileNotFoundError:
            
            print("The model '"+self.model_dir+"' does not exist.")
            
            if only_load:
                raise ValueError("The model directory was not created as 'only_load' was set to True.")
            
            else:
                io_util.save_makedirs(self.model_dir)
                print("The model was created at '"+self.model_dir+"'.")        
        
        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.tile_size = tile_size
        self.num_channels = num_channels
        
        misc.print_line()
        
        
    def init(self, hyperparameters = None):
        '''
        Initialize the model. Note that this is different from self.__init__ in that it will create a new model. Note that the attributes model_id, hyperparameters, num_channels and tile_size must be defined, or a some error might occur. If any of the attributes are missing, define them using, eg. self.num_channels = 3.
        
            Options:
                hyperparameters - dict - Optional - If provided, then self.hyperparameters will be changed.
        '''
        
        if not hyperparameters is None:
            self.hyperparameters = hyperparameters
        
        self.model = init_model(
            #model_id = self.model_id, 
            tile_size = self.tile_size, 
            num_channels = self.num_channels, 
            hyperparameters = self.hyperparameters,
        )
        
        self.save()
    
        
    def summary(self):
        return self.model.summary     
        
        
    def load(self, load_pickle = False):
        '''
        Load the model contained in 'self.model_dir', using the pickled version if 'Model.pickle' exists, and the meta info ***FIXME: elaborate*** otherwise.
        
        Options:
            load_pickle - bool - If True, loads the attributes of self (other than self.model) from Model.attributes.pickle.
        
        Raises:
            FileNotFoundError: If self.model_dir does not exist.
        '''
        
        if os.path.exists(self.model_dir):
                  
            # Attempt to load the pickle file if pickle = True. The pickle file should contain a dictionary of the Model attributes (other than Model.model).
            if load_pickle:
                
                picklefile = self.model_dir+"/Model.attributes.pickle"
                
                try:
                    with open(picklefile, "rb") as input_file:
                        
                        attributes = pickle.load(input_file)
                        for attr, value in attributes.items():
                            setattr(self, attr, value)
                        
                except FileNotFoundError:
                    print("No pickle file was found at '" + picklefile + "'. The Model attributes were not loaded.'")
                    pass
        
            self.load_model()
        
        else:
            raise FileNotFoundError('The model '+self.model_dir+' does not exist.')
        
        misc.print_line()
        
    
    def unload_model(self):
        '''
        The model self.model can take up quite a bit of memory. This function can be used to unload it.
        '''
        
        import gc
        
        del self.model
        
        gc.collect()
        
    def load_model(self):
        '''
        This loads the model in self.model_dir into self.model.
        '''
        
        self.model = io_util.load_model(self.model_id, self.model_dir)
        
                  
    def save(self, save_pickle = False):
        '''
        Save self to self.model_dir, creating any necessary directories. After a successful save, model_dir will contain some meta info ***FIXME: elaborate*** from the Sequential object, and a pickled Model (this) object.
        
        Options:
            save_pickle - bool - If True, saves the attributes of self (other than self.model) to Model.attributes.pickle.
        '''
        
        #io_util.save_makedirs(self.model_dir)
        
        # Save the model meta info ***FIXME: elaborate***.
        io_util.save_makedirs(self.model_dir)
        io_util.save_model(self.model, self.model_dir)
        io_util.save_model_summary(self.hyperparameters, self.model, self.model_dir)
        
        # Pickling is currently not functioning properly.
        if save_pickle:

            # Also export a pickle Model that saves the attributes.
            
            picklefile = self.model_dir+"/Model.attributes.pickle" #+pickle.HIGHEST_PROTOCOL
            
            with open(picklefile, "wb") as output_file:
                
                # self.model is saved separately. Attempting to save it as well raises a 'TypeError: can't pickle _thread.RLock objects'.
                pickle.dump(self.attributes(exclude = {'model'}), output_file, pickle.HIGHEST_PROTOCOL) 
                
                print("The attributes of the Model object were saved to '" + picklefile + "'.")

        print("The model was saved to '" + self.model_dir + "'.")
        misc.print_line()
        
        
    def compile(self):
        self.model = compile_model(self.model, self.hyperparameters["learning_rate"], self.hyperparameters["momentum"], self.hyperparameters["decay"])
        
        self.save()

    def train(self, features_train, labels_train, epochs = 100, checkpoints = False, tensorboard = False):
        self.model, history = train_model(
            self.model,
            features_train,
            labels_train,
            tile_size = self.tile_size,
            model_id = self.model_id,
            epochs=epochs,
            checkpoints = checkpoints,
            tensorboard = tensorboard
        )
        
        #print(history.history['loss'][-1])
        
        self.save()
        
        return history

    def evaluate(self, features_test, labels_test, out_format = "GeoTIFF"):
        evaluate_model(
            self.model, 
            features_test, 
            labels_test, 
            self.tile_size,
            self.model_dir, 
            out_format
        )
        
        
    def attributes(self, exclude = {}):
        '''
        Returns all the attributes of self, except those in exclude.
        
        Options:
            exclude - set - A set of attributes (strings) to ignore.
        
        Returns:
            A dict of attributes.
        '''
        
        attributes = dict()
        
        for attr, value in self.__dict__.items():
            if attr not in exclude:
                attributes[attr] = value
                
        return attributes
    
        
    def spawn_child(self, excluded_attributes = {}, subdir = 'submodels'):
        '''
        Returns a Model that has the same attributes as self, except those in excluded_attributes ('model_id', 'model_dir' are always excluded). The Model also gets the additional parent_model_id and parent_models_dir attributes.
        
        Options:
            excluded_attributes - set of str - Attributes to ignore during the copying.  
        '''
        
        child = Model(
            model_id = self.model_id + '_' + str(time.strftime("%d_%m_%Y_%H%M%S")),
            models_dir = os.path.join(self.model_dir, subdir) 
        )
        
        exclude = {'model_id', 'model_dir'}.union(excluded_attributes)
        
        for attr, value in self.attributes(exclude = exclude).items():
            setattr(child, attr, copy.deepcopy(value))
        
        #child.generate_model_id()
        
        child.parent_model_id = self.model_id
        child.parent_model_dir = self.model_dir
        
        print('A child model was spawned with the following attributes: ')
        print(child.attributes())
        misc.print_line()
        
        return child 
        
    
    #def loss