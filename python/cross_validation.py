__all__ = ['CV_Skeleton']

import numpy as np

from keras import backend as K
from keras.layers import Dense

class CV_Skeleton:

    def __init__(self, in_X, in_y, tuning_params, preproc_method, cv_method, ml_model, categorial_y=True):

        self.in_X           = in_X
        self.in_y           = in_y
        self.tuning_params  = tuning_params
        self.preproc_method = preproc_method
        self.cv_method      = cv_method
        self.ml_model       = ml_model
        self.cv_dict        = {}
        self.cv_dict['model']        = self.ml_model
        if categorial_y:
            from keras.utils import to_categorical
            self.sparse_y   = to_categorical(self.in_y)

    def get_X(self):
        return self.in_X
    def get_y(self):
        return self.in_y

    def create_train_test_indices(self):
        return list(self.cv_method.split(self.in_X, self.in_y))
    def get_train_test_indices(self):
        return self.train_test_indices

    def get_preproc_params(self):
        return self.preproc_method.get_params()

    def reset_model(self):
        session = K.get_session()
        for layer in self.ml_model.layers:
            if isinstance(layer, Dense):
                old = layer.get_weights()
                layer.weights[0].initializer.run(session=session)
                layer.weights[1].initializer.run(session=session)
            else:
                print(layer, "not reinitialized")

    def get_cv_dict(self):
        return self.cv_dict

    def cv_loop(self):
        self.train_test_indices = self.create_train_test_indices()
        for idx, ifold in enumerate(self.train_test_indices):
            best_loss = 100.0
            train_id, test_id = ifold[0], ifold[1]
            self.cv_dict[idx] = {}
            self.cv_dict[idx]['trn_tst_indices'] = ifold
            for init in range(self.tuning_params['n_inits']):
                print('Training in the Fold: %i | Init: %i' %(idx+1, init+1))
                # reset the weights of model
                self.reset_model()
                # create a scaler to prepare the data and fit using only the train data
                self.preproc_method.fit(self.in_X[train_id])
                # transform all data
                self.X_norm = self.preproc_method.transform(self.in_X)

                self.train_evo = self.ml_model.fit(self.X_norm[train_id], self.sparse_y[train_id],
                              batch_size=self.tuning_params['batch_size'],
                              epochs=self.tuning_params['epochs'],
                              validation_data=(self.X_norm[test_id], self.sparse_y[test_id]))

                if np.min(self.train_evo.history['val_loss']) < best_loss:
                    best_loss = np.min(self.train_evo.history['val_loss'])
                    self.cv_dict[idx]['history'] = self.train_evo.history
                    self.cv_dict[idx]['weights'] = self.ml_model.get_weights()
