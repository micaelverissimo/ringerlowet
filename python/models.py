__all__ = ['create_mlp_model']

import numpy as np
import tensorflow as tf

def create_mlp_model(input_dim, output_dim,
                        n_neurons,
                        hl_act_func = tf.nn.relu,
                        ol_act_func = tf.nn.softmax,
                        n_layers=1,
                        loss_func = 'mean_squared_error',
                        optimizer = 'adam',
                        metrics = None,
                        name='model'):
    # Create model
    model = tf.keras.models.Sequential(name=name)
    for ilayer in range(n_layers):
        model.add(tf.keras.layers.Dense(n_neurons if type(n_neurons) != type([]) else n_neurons[ilayer],
                        input_dim=input_dim, activation=hl_act_func, name='layer_%i' %(ilayer+1) ))
    model.add(tf.keras.layers.Dense(output_dim, activation=ol_act_func, name='output_layer'))

    # Compile model
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics= metrics if metrics != None else ['accuracy'])
    return model
