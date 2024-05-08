import pandas as pd
import tensorflow as tf
from tensorflow import feature_column

def file_reader():
    data = pd.read_csv('~\\UNSW_NB15_training-set.csv')
    data_ = pd.read_csv('~\\UNSW_NB15_testing-set.csv')
    data = pd.concat([data, data_], ignore_index=True)
    return data

def data_preparation(data):
    data_features = data
    if set(['id','attack_cat','label']).issubset(set(data_features.columns)):
        data_features = data_features.drop(['id','attack_cat','label'], axis=1)
    if set(['label']).issubset(set(data.columns)):
        data_labels = data.pop('label')
    
    return data_features,data_labels

def  shuffle_dataset(data_features,data_labels,batch_size,shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((dict(data_features),data_labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_features))
    ds = ds.batch(batch_size)
    return ds

def column_transformation(training_data_features,dataframe):
    feature_columns = []
    categorical_columns = []
    numeric_columns =  []
    float_columns = []
    dtypes = training_data_features.dtypes.to_dict()
    for column,type in dtypes.items():
        if type == 'float64':
            float_columns.append(column)
        elif type == 'int64':
            numeric_columns.append(column)
        elif type == 'O':
            categorical_columns.append(column)
    for column_name in numeric_columns:
        feature_columns.append(feature_column.numeric_column(column_name))
    for column_name in float_columns:
        feature_columns.append(feature_column.numeric_column(column_name))
    for column_name in categorical_columns:
        categorical_column = feature_column.categorical_column_with_vocabulary_list(column_name, dataframe[column_name].unique())
        indicator_column = feature_column.indicator_column(categorical_column)
        feature_columns.append(indicator_column)
    return feature_columns
