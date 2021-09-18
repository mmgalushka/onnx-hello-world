import os
import json
import zipfile
import pandas as pd
import numpy as np

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

from PIL import Image
from matplotlib import pyplot as plt


class Schema:
    """The data schema.

    A data descriptor is a JSON object which lists the data features their
    kinds and defines which feature acts as the target. For example:

        {
            "features": [
                {
                    "name": "Survived",
                    "kind": "categorical"
                },
                {
                    "name": "Pclass",
                    "kind": "categorical"
                },
                {
                    "name": "Name"
                },
                {
                    "name": "Sex",
                    "kind": "categorical"
                },
                {
                    "name": "Age",
                    "kind": "numeric"
                },
                {
                    "name": "Siblings and Spouses Aboard"
                },
                {
                    "name": "Parents and Children Aboard"
                },
                {
                    "name": "Fare",
                    "kind": "numeric"
                }
            ],
            "target": 0
        }

    This descriptor includes 8 features. Features containing the attribute
    "kind" are considered for creating the schema (in this example these
    features are Survived, Pclass, Sex, Age, and Fare).  Features Survived,
    Pclass, and Sex are categorical. Features Age and Fare are numeric. The
    "target" defines the index of the prediction feature (in this example
    index is 0, so the feature Survived will be the target).

    Args:
        descriptor (dict): The data descriptor.
    """

    def __init__(self, descriptor):
        self.__numeric_features = []
        self.__categorical_features = []
        self.__target = None

        for idx, feature in enumerate(descriptor['features']):
            if idx == descriptor['target']:
                self.__target = feature['name']
            else:
                if 'kind' in feature:
                    if feature['kind'] == 'numeric':
                        self.__numeric_features.append(feature['name'])
                    elif feature['kind'] == 'categorical':
                        self.__categorical_features.append(feature['name'])

    @property
    def numeric_features(self):
        """Returns the list of numeric features."""
        return self.__numeric_features

    @property
    def categorical_features(self):
        """Returns the list of categorical features."""
        return self.__categorical_features

    @property
    def features(self):
        """Returns the list of training features [numeric + categorical]."""
        return self.numeric_features + self.categorical_features

    @property
    def target(self):
        """Returns the index of the target feature."""
        return self.__target


class Dataset:
    """A dataset handler.

    This class intends to simplify some routine operations with the dataset.

    Args:
        schema (Schema): The data schema.
        data (DataFrame): The data frame.
    """

    def __init__(self, schema, data):
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
        """Returns the data schema."""
        return self.__schema

    @property
    def X(self):
        """Returns training data numeric + categorical features."""
        return self.__data[self.__schema.features]

    @property
    def y(self):
        """Returns the target feature."""
        return self.__data[self.__schema.target]

    def train_test_split(self):
        """Splits data on training and test (test_size=0.33)."""
        train, test = train_test_split(
            self.__data, test_size=0.33, random_state=42
        )
        return Dataset(self.schema, train), Dataset(self.schema, test)


def load_dataset(dataset_name):
    """Loads dataset by name.

    The loading dataset must be zipped and located in teh "data" directory.

    Args:
        dataset_name (str): The dataset to load.
    """
    archive = zipfile.ZipFile(f'data/{dataset_name}.zip', 'r')
    with archive.open('descriptor.json') as f:
        descriptor = json.load(f)
    with archive.open('data.csv') as f:
        data = pd.read_csv(f)
    return Dataset(Schema(descriptor), data)


def create_preprocessor(dataset):
    """Creates the column transformer from dataset.

    Args:
        dataset (Dataset): The data set to use for creating the column
            transformer.

    Returns:
        (ColumnTransformer): The created column transformer

    """
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


def load_image(dataset_name, image_name):
    """Loads an image.

    The loading dataset must be zipped and located in teh "data" directory.

    Args:
        dataset_name (str): The dataset to load.
        image_name (str): The image to load (stored inside loaded zip).
    """
    with zipfile.ZipFile(f'data/{dataset_name}.zip', 'r') as a:
        with a.open(f'{image_name}.jpeg', 'r') as f:
            return Image.open(f).copy()


def show_image(image, model_name, image_category, image_name, dataset_name):
    """Show an image and saves it to the file.

    The login for generating file name:
    >>> if image_category == 'orig':
    >>>    filename = f'{image_category}_{image_name.replace("-","_")}.jpg'
    >>> else:
    >>>    filename = f'{model_name.lower()}_{image_category}_{image_name.replace("-","_")}.jpg'

    Args:
        image (obj): The image to show;
        model_name (str): The model name (ex. 'MaskRCNN').
        image_category (str): The image category (can be 'orig', 'pre', 'post').
        image_name (str): The image name (ex. cat-1).
        dataset_name (str): The dataset name (ex. 'animal').
    """
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
    """Returns the ONNX types from the dataset.

    Args:
        dataset (Dataset): The data set to use for creating ONNX types.
        drop (list): The list of features to drop.

    Returns:
        (list) The list of tuples defining the ONNX model input types.
    """
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
    """Transforms the Dataset object to the ONNX model input.

    Args:
        dataset (Dataset): The data set to use for creating ONNX input.

    Returns:
        (dict) The dictionary defining the ONNX model input.
    """
    return {
        column: dataset.X[column].values.reshape(-1, 1)
        for column in dataset.X.columns
    }
