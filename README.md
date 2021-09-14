# ONNX Hello World

![Project Logo](/logo.png)

[![Project License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/mmgalushka/bootwrap/blob/main/LICENSE)

This project provides educational material for using [Open Neural Network Exchange (ONNX)](https://onnx.ai/). These materials would be useful for data scientists and engineers who are planning to use the ONNX ML models in their AI projects. Notebooks will help you to find answers to the following questions: 
* How I can convert my ML classification [SKLearn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.readthedocs.io/en/latest/), and [Tensorflow (Keras)](https://www.tensorflow.org/) models into the ONNX format? 
* What is the difference between the original ML model and the model converted into the ONNX?
* Where I can find and how I can use already trained and serialized ONNX models?
* How I can visualize the ONNX graph?

**Note:** In notebooks, we will not spend time performing data cleaning, feature selection, model optimization, and etc. This project is not about building the best models, it is about ONNX!

## What is ONNX?

Let me cite a quote defining the goal of the ONNX project defined [here](https://onnx.ai/about.html):

> "Our goal is to make it possible for developers to use the right combinations of tools for their project. We want everyone to be able to take AI from research to reality as quickly as possible without artificial friction from toolchains."

If you would like about ONNX please follow the following [link](https://github.com/onnx/).

You might be asking the question: _why do we need this project if we already have excellent documentation [here](https://github.com/onnx/)_? I believe the best way to learn something new is to try-examples-yourself. So I tried to follow the existing documentation, repeat, and introduce some new steps. This way I tried to learn this technology and share my experience with others.

## Experiments

| Descriptions                           | Notebook | Summary |
| -------------------------------------- | -------- | ------- |
| [SKLearn](exp/sklearn.md)              | [onnx_sklearm.ipynb](onnx_sklearm.ipynb) | SKLearn models training, conversion, and comparing to ONNX |
| [XGBoost](exp/xgboost.md)              | [onnx_xgboost.ipynb](onnx_xgboost.ipynb) | XGBoost model training, conversion, and comparing to ONNX |
| [Tensorflow(Keras)](exp/tensorflow.md) | [onnx_tensorflow.ipynb](onnx_tensorflow.ipynb) | Tensorflow(Keras) model training, conversion, and comparing to ONNX |
| [Resnet](exp/tensorflow.md)            | [onnx_resnet.ipynb](onnx_resnet.ipynb) | Inference using ResNet ONNX  |
| [MaskRCNN](exp/maskrcnn.md)            | [onnx_maskrcnn.ipynb](onnx_maskrcnn.ipynb) | Inference using MaskRCNN ONNX |

## Links

* [ONNX Official Website](https://onnx.ai/)
* [ONNX GitHub](https://github.com/onnx)
* [Experimental Notebooks](https://github.com/mmgalushka/onnx-hello-world)
* [Issue Tracker](https://github.com/mmgalushka/onnx-hello-world/issues)
