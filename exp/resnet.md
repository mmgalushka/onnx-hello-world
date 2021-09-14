# Experiment with ResNet Model

To run experiments with the [Resnet](https://arxiv.org/abs/1512.03385) model use [onnx_resnet.ipynb](../onnx_resnet.ipynb) notebook. Using this notebook we conducted experiments with the pre-trained ResNet classifier.

## Dataset

For prediction experiments, we are using an "animal" data set. This dataset combines photographs cat(s) and dog(s) collected from the internet.

## Prediction Results

| Animal      | Original Image                       | Preprocessed Image                                      | Label                  | Score |
| ----------- | ------------------------------------------- | ------------------------------------------------ | ---------------------- | ----- |
| cat-1       | ![resnet orig cat 1](images/orig_cat_1.jpg) | ![resnet_pre_cat_1](images/resnet_pre_cat_1.jpg) | "tabby, tabby cat"     | 12.97 |
| cat-2       | ![resnet orig cat 2](images/orig_cat_2.jpg) | ![resnet_pre_cat_2](images/resnet_pre_cat_2.jpg) | "tabby, tabby cat"     | 7.93  |
| cat-3       | ![resnet orig cat 3](images/orig_cat_3.jpg) | ![resnet_pre_cat_3](images/resnet_pre_cat_3.jpg) | "Egyptian cat"         | 8.87  |
| dog-1       | ![resnet orig dog 1](images/orig_dog_1.jpg) | ![resnet_pre_dog_1](images/resnet_pre_dog_1.jpg) | "Labrador retriever"   | 14.60 |
| dog-2       | ![resnet orig dog 2](images/orig_dog_2.jpg) | ![resnet_pre_dog_2](images/resnet_pre_dog_2.jpg) | "Bernese mountain dog" | 13.96 |
| dog-3       | ![resnet orig dog 3](images/orig_dog_3.jpg) | ![resnet_pre_dog_3](images/resnet_pre_dog_3.jpg) | "Eskimo dog, husky"    | 11.79 |
