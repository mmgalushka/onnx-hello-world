import os
import json
import zipfile
import pandas as pd
import numpy as np

from PIL import Image

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from skl2onnx.common.data_types import (
    Int64TensorType,
    FloatTensorType,
    DoubleTensorType,
    StringTensorType
)

from matplotlib import pyplot as plt


class Schema:
    def __init__(self, descriptor):
        self.__numeric_features = []
        self.__categorical_features = []
        self.__target = None
        self.__labels = None

        for idx, feature in enumerate(descriptor['features']):
            if idx == descriptor['target']:
                self.__target = feature['name']
            else:
                if 'kind' in feature:
                    if feature['kind'] == 'numeric':
                        self.__numeric_features.append(feature['name'])
                    elif feature['kind'] == 'categorical':
                        self.__categorical_features.append(feature['name'])

        if 'labels' in descriptor:
            self.__labels = descriptor['labels']

    @property
    def numeric_features(self):
        return self.__numeric_features

    @property
    def categorical_features(self):
        return self.__categorical_features

    @property
    def features(self):
        return self.numeric_features + self.categorical_features

    @property
    def target(self):
        return self.__target

    @property
    def labels(self):
        return self.__labels


class Dataset:
    def __init__(self, schema, data, labels=None):
        self.__schema = schema
        self.__data = data

    def __str__(self):
        return str(self.__data)

    def __setitem__(self, feature, data):
        self.__data[feature] = data

    def __getitem__(self, feature):
        return self.__data[feature]

    def __len__(self):
        return len(self.__data)

    @property
    def schema(self):
        return self.__schema

    @property
    def X(self):
        return self.__data[self.__schema.features]

    @property
    def y(self):
        return self.__data[self.__schema.target]

    def train_test_split(self):
        train, test = train_test_split(
            self.__data, test_size=0.33, random_state=42
        )
        return Dataset(self.schema, train), Dataset(self.schema, test)


def load_dataset(dataset_name):
    archive = zipfile.ZipFile(f'data/{dataset_name}.zip', 'r')
    with archive.open('descriptor.json') as f:
        descriptor = json.load(f)
    with archive.open('data.csv') as f:
        data = pd.read_csv(f)
    return Dataset(Schema(descriptor), data)


def load_image(dataset_name, image_name):
    with zipfile.ZipFile(f'data/{dataset_name}.zip', 'r') as a:
        with a.open(f'{image_name}.jpeg', 'r') as f:
            return Image.open(f).copy()


def show_image(image, model_name, image_category, image_name, dataset_name):
    _, ax = plt.subplots()
    plt.imshow(image, interpolation='nearest')

    if image_category == 'orig':
        image_label = 'original'
    elif image_category == 'pre':
        image_label = 'preprocessed'
    elif image_category == 'post':
        image_label = 'postprocessed'
    else:
        image_label = ''

    ax.set_title(
        f'The {image_label} image of "{image_name}" from\nthe "{dataset_name}" dataset'
    )

    if image_category == 'orig':
        filename = f'{image_category}_{image_name.replace("-","_")}.jpg'
    else:
        filename = f'{model_name.lower()}_{image_category}_{image_name.replace("-","_")}.jpg'

    plt.savefig(os.path.join('tmp', filename))
    plt.show()


def get_onnx_input_type(dataset, drop=None):
    input_type = []
    for k, v in zip(dataset.X.columns, dataset.X.dtypes):
        if drop is not None and k in drop:
            continue
        if v == 'int64':
            t = Int64TensorType([None, 1])
        elif v == 'float32':
            t = FloatTensorType([None, 1])
        elif v == 'float64':
            t = DoubleTensorType([None, 1])
        else:  # v == object
            t = StringTensorType([None, 1])
        input_type.append((k, t))
    return input_type


def get_onnx_input_data(dataset):
    return {
        column: dataset.X[column].values.reshape(-1, 1)
        for column in dataset.X.columns
    }


def create_preprocessor(dataset):
    # The numeric features preprocessing pipeline.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # The categorical features preprocessing pipeline.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # The data preprocessor combines numeric and categorical
    # column transformers.
    transformers = []
    if len(dataset.schema.numeric_features) > 0:
        transformers.append(('num', numeric_transformer,
                            dataset.schema.numeric_features))
    if len(dataset.schema.categorical_features) > 0:
        transformers.append(('cat', categorical_transformer,
                            dataset.schema.categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor
