__all__ = ['create_mlp_model']

import numpy as np

from keras.models import Sequential
from keras.layers import Dense

def create_mlp_model(input_dim, output_dim,
                        n_neurons,
                        hl_act_func = 'relu',
                        ol_act_func = 'softmax',
                        n_layers=1,
                        loss_func = 'mean_squared_error',
                        optimizer = 'adam',
                        metrics = None,
                        name='model'):
    # Create model
    model = Sequential(name=name)
    for ilayer in range(n_layers):
        model.add(Dense(n_neurons if type(n_neurons) != type([]) else n_neurons[ilayer],
                        input_dim=input_dim, activation=hl_act_func))
    model.add(Dense(output_dim, activation=ol_act_func))

    # Compile model
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics= metrics if metrics != None else ['accuracy'])
    return model
