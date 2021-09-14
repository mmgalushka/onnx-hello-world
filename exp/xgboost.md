# Experiment with XGBoost Model

To run experiments with the [XGBoost](https://xgboost.readthedocs.io/en/latest/) model use [onnx_xgboost.ipynb](../onnx_xgboost.ipynb) notebook. Using this notebook we conducted experiments with the XGBoost classifier.

## Dataset

For performing experiments, we will be using the "iris" classification data set. This dataset describes three species of Iris flower using four numeric features presented in the following table:

| Feature      | Kind        | Target             |
| -------------| ----------- | ------------------ |
| sepal_length | numeric     | :x:                |
| sepal_width  | numeric     | :x:                |
| petal_length | numeric     | :x:                |
| petal_width  | numeric     | :x:                |
| class        | categorical | :heavy_check_mark: |

The "class" field defines three species of Iris (Iris **setosa**, Iris **virginica** and Iris **versicolor**).

## Models Comparison Results

The results of conducted experiments are presented in the following table.

| Cassifier               | Original | ONNX | Probabilities Difference                 |
| ----------------------- | -------- | ---- | ---------------------------------------- |
| XGBoost                 | 100%     | 100% | ![diff_xgboost](images/diff_xgboost.jpg) |

## Summary

Overall the process of working with the XGBoost classifier is very similar to the SKLearn model. You can easily observe this just by comparing both notebooks. The key difference is in performing Registration for the XGBoost convertor:

```Python
from xgboost import XGBClassifier
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

...
update_registered_converter(
    XGBClassifier, 'XGBoostXGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

onnx_model = convert_sklearn(model, initial_types=...)    
...
```

By observing experimental results, original and ONNX models have very similar behavior with **tiny** differences in prediction probability for outliers.
