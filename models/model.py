import tensorflow as tf
from tensorflow.keras import layers


def model_a(feature_layer,initial_bias):
    model = tf.keras.Sequential()
    model.add(feature_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16, kernel_initializer = 'uniform' ,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, kernel_initializer = 'uniform' ,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, kernel_initializer = 'uniform' ,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid',bias_initializer = tf.keras.initializers.Constant(initial_bias)))

    return model


def model_b(feature_layer,initial_bias):
    model = tf.keras.Sequential()
    model.add(feature_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16, kernel_initializer = 'uniform' ,kernel_regularizer = tf.keras.regularizers.l2(0.01), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32,kernel_initializer = 'uniform', kernel_regularizer = tf.keras.regularizers.l2(0.01), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64,kernel_initializer = 'uniform', kernel_regularizer = tf.keras.regularizers.l2(0.01), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128,kernel_initializer = 'uniform', kernel_regularizer = tf.keras.regularizers.l2(0.01), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid',bias_initializer = tf.keras.initializers.Constant(initial_bias)))

    return model



def model_c(feature_layer,initial_bias):
    model = tf.keras.Sequential()
    model.add(feature_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid',bias_initializer = tf.keras.initializers.Constant(initial_bias)))

    return model


def model_d(feature_layer,initial_bias):
    model = tf.keras.Sequential()
    model.add(feature_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid',bias_initializer = tf.keras.initializers.Constant(initial_bias)))

    return model



def model_e(feature_layer,initial_bias):
    model = tf.keras.Sequential()
    model.add(feature_layer)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(16,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128,kernel_regularizer = tf.keras.regularizers.l2(0.02), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid',bias_initializer = tf.keras.initializers.Constant(initial_bias)))

    return model
