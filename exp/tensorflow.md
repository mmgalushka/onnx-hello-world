# Experiment with Tensorflow(Keras) Model

To run experiments with the [Tensorflow(Keras)](https://www.tensorflow.org/) model use [onnx_tensorflow.ipynb](onnx_tensorflow.ipynb) notebook. Using this notebook we conducted experiments with the Tensorflow(Keras) classifier.

## Dataset

For performing experiments, we will be using the "adult" classification data set. This dataset is based on the Census Bureau and used for predicting whether a given adult makes more than $50,000 based on features presented in the following table.

| Feature      | Kind        | Target             |
| -------------| ----------- | ------------------ |
| Age          | numeric     | :x:                |
| fnlwgt       | numeric     | :x:                |
| EducationNum | numeric     | :x:                |
| CapitalGain  | numeric     | :x:                |
| CapitalLoss  | numeric     | :x:                |
| HoursPerWeek | numeric     | :x:                |
| WorkClass    | categorical | :x:                |
| Education    | categorical | :x:                |
| MaritalStatus| categorical | :x:                |
| Occupation   | categorical | :x:                |
| Relationship | categorical | :x:                |
| Race         | categorical | :x:                |
| Gender       | categorical | :x:                |
| NativeCountry| categorical | :x:                |
| Income       | categorical | :heavy_check_mark: |

The "Income" field defines two income categories: **<=50K** and **>50K**.

## Models Comparison Results

The results of conducted experiments are presented in the following table.

| Cassifier               | Original | ONNX | Probabilities Difference                          |
| ----------------------- | -------- | ---- | ------------------------------------------------- |
| Tensorflow(Keras)       | 86%      | 86%  | ![diff_tensorflow](images/diff_tensorflow.jpg)    |

## Summary

A conversation of the Tensorflow(Keras) model into ONNX format is slightly different from [SKLearn](onnx_sklearn.ipynb) and [XGBoost](onnx_xgboost.ipynb). The main difference lies in preprocessing data. It must be done "manually" before feeding data to the classifier. For the actual model conversion, we tried to use [keras-onnx](https://github.com/onnx/keras-onnx) packages. Unfortunately, the result was unsuccessful. As a workaround, we save the original model in the file using Tensorflow format and convert it into the ONNX using [tf2onnx](https://github.com/onnx/tensorflow-onnx) package.

```Python
import tf2onnx.convert

...
model.save(
    os.path.join('tmp', 'model.h5'),
    overwrite=True, include_optimizer=False, save_format='tf'
)
...
model = keras.models.load_model(os.path.join('tmp', 'model.h5'))
onnx_model, _ = tf2onnx.convert.from_keras(model)
```

By observing experimental results, original and ONNX models have very similar behavior with **tiny** differences in prediction probability for outliers.
