import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def model_a(feature_layer,init,activation,dropout_rate,regularizer = tf.keras.regularizers.l2(0.02)):
    model = tf.keras.Sequential()
    model.add(feature_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16, kernel_initializer = init ,kernel_regularizer = regularizer, activation = activation))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(32, kernel_initializer = init ,kernel_regularizer = regularizer, activation = activation))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, kernel_initializer = init ,kernel_regularizer = regularizer, activation = activation))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    return modela


def model_b(feature_layer,init,activation,dropout_rate,regularizer = tf.keras.regularizers.l2(0.02)):
    model = tf.keras.Sequential()
    model.add(feature_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16, kernel_initializer = init ,kernel_regularizer = regularizer, activation = activation))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(32,kernel_initializer = init, kernel_regularizer = regularizer, activation = activation))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64,kernel_initializer = init, kernel_regularizer = regularizer, activation = activation))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    return modelb


def gridSearchParams(model,X,Y):
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    activation = ['relu','tanh']
    param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint, activation=activation,init_mode=init_mode)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(self.X, self.Y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    


def main():
    model_a = KerasClassifier(build_fn = model_a, epochs=10, batch_size=32, verbose=0)
    model_b = KerasClassifier(build_fn = model_b, epochs=10, batch_size=32, verbose=0)
    for model in [model_a,model_b]:
        gridSearchParams(model,X,Y)


if __name__ == "__main__":
    main()


