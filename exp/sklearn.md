# Experiments with SKLearn Models

To run experiments with the [SKLearn](https://scikit-learn.org/stable/) models use [onnx_sklearn.ipynb](../onnx_sklearn.ipynb) notebook. Using this notebook we conducted experiments with six SKLearn classifiers: K-nearest neighbors, Logistic Regression, Random Forest, Support Vector, Gaussian Naive Bayes, and Multi-layer Perceptron.

## Dataset

For performing experiments, we will be using the "titanic" classification data set. This dataset describes the survival status of individual passengers on the Titanic. It contains a combination of numeric and categorical features showing in the following table.

| Feature      | Kind        | Target             |
| ------------ | ----------- | ------------------ |
| Pclass       | categorical | :x:                |
| Sex          | categorical | :x:                |
| Age          | numeric     | :x:                |
| Fare         | numeric     | :x:                |
| Survived     | categorical | :heavy_check_mark: |

The "Survived" field defines a passenger survived status (**0**-not survived and **1**-survived).

This combination of features is useful for testing the data preprocessing pipeline.

## Models Comparison Results

The results of conducted experiments are presented in the following table.

| Cassifier               | Original | ONNX | Probabilities Difference            |
| ----------------------- | -------- | ---- | ----------------------------------- |
| K-nearest neighbors     | 81%      | 81%  | ![diff_knn](images/diff_knn.jpg)    |
| Logistic Regression     | 77%      | 77%  | ![diff_lr](images/diff_lr.jpg)      |
| Random Forest           | 81%      | 81%  | ![diff_rf](images/diff_rf.jpg)      |
| Support Vector          | 77%      | 77%  | ![diff_svm](images/diff_svm.jpg)    |
| Gaussian Naive Bayes    | 75%      | 75%  | ![diff_nb](images/diff_nb.jpg)      |
| Multi-layer Perceptron  | 77%      | 77%  | ![diff_mlp](images/diff_mlp.jpg)    |

## Summary

All of our tested classifiers were successfully converted to the ONNX format. The ONNX models produced the same accuracy results as the correspondent SKLearn models. Very similar behavior (according to the difference in prediction probability) showed Logistic Regression, Support Vector, Gaussian Naive Bay, and Multi-layer Perceptron classifiers. K-nearest neighbors and Random Forest classifiers showed surprisingly large differences in prediction probabilities for some samples. This potentially may cause a prediction swing to other classes.