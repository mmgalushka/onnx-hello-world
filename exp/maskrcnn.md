# Experiment with MaskRCNN Model

To run experiments with the [MaskRCNN](https://arxiv.org/abs/1703.06870) model use [onnx_maskrcnn.ipynb](../onnx_maskrcnn.ipynb) notebook. Using this notebook we conducted experiments with the Mask-RCNN deep neural network model aimed to solve instance segmentation problem in computer vision.

## Dataset

For prediction experiments, we are using an "animal" data set. This dataset combines photographs cat(s) and dog(s) collected from the internet.

| Animal    | Original Image                               | Preprocessed Image                                           | Postprocessed Image                                            |
| ----------| -------------------------------------------- |------------------------------------------------------------- | ---------------------------------------------------------------|
| cat-1     | ![orig_cat_1](images/orig_cat_1.jpg)         | ![maskrcnn_pre_cat_1](images/maskrcnn_pre_cat_1.jpg)         | ![maskrcnn_post_cat_1](images/maskrcnn_post_cat_1.jpg)         |
| dog-1     | ![orig_dog_1](images/orig_dog_1.jpg)         | ![maskrcnn_pre_dog_1](images/maskrcnn_pre_dog_1.jpg)         | ![maskrcnn_post_dog_1](images/maskrcnn_post_dog_1.jpg)         |
| cat-n     | ![orig_cat_n](images/orig_cat_n.jpg)         | ![maskrcnn_pre_cat_n](images/maskrcnn_pre_cat_n.jpg)         | ![maskrcnn_post_cat_n](images/maskrcnn_post_cat_n.jpg)         |
| dog-n     | ![orig_dog_n](images/orig_dog_n.jpg)         | ![maskrcnn_pre_dog_n](images/maskrcnn_pre_dog_n.jpg)         | ![maskrcnn_post_dog_n](images/maskrcnn_post_dog_n.jpg)         |
| dog-n-cat | ![orig_dog_n_cat](images/orig_dog_n_cat.jpg) | ![maskrcnn_pre_dog_n_cat](images/maskrcnn_pre_dog_n_cat.jpg) | ![maskrcnn_post_dog_n_cat](images/maskrcnn_post_dog_n_cat.jpg) |