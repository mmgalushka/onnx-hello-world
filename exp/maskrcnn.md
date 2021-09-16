# Experiment with MaskRCNN Model

To run experiments with the [MaskRCNN](https://arxiv.org/abs/1703.06870) instance segmentation model use [onnx_maskrcnn.ipynb](../onnx_maskrcnn.ipynb) notebook.

## Dataset

For prediction experiments, we are using an "animal" data set. This dataset combines photographs cat(s) and dog(s) collected from the internet.

## Prediction Results

| Animal    | Original Image                               | Preprocessed Image                                           | Postprocessed Image                                            |
| ----------| -------------------------------------------- |------------------------------------------------------------- | ---------------------------------------------------------------|
| cat-1     | ![orig_cat_1](images/orig_cat_1.jpg)         | ![maskrcnn_pre_cat_1](images/maskrcnn_pre_cat_1.jpg)         | ![maskrcnn_post_cat_1](images/maskrcnn_post_cat_1.jpg)         |
| dog-1     | ![orig_dog_1](images/orig_dog_1.jpg)         | ![maskrcnn_pre_dog_1](images/maskrcnn_pre_dog_1.jpg)         | ![maskrcnn_post_dog_1](images/maskrcnn_post_dog_1.jpg)         |
| cat-n     | ![orig_cat_n](images/orig_cat_n.jpg)         | ![maskrcnn_pre_cat_n](images/maskrcnn_pre_cat_n.jpg)         | ![maskrcnn_post_cat_n](images/maskrcnn_post_cat_n.jpg)         |
| dog-n     | ![orig_dog_n](images/orig_dog_n.jpg)         | ![maskrcnn_pre_dog_n](images/maskrcnn_pre_dog_n.jpg)         | ![maskrcnn_post_dog_n](images/maskrcnn_post_dog_n.jpg)         |
| dog-n-cat | ![orig_dog_n_cat](images/orig_dog_n_cat.jpg) | ![maskrcnn_pre_dog_n_cat](images/maskrcnn_pre_dog_n_cat.jpg) | ![maskrcnn_post_dog_n_cat](images/maskrcnn_post_dog_n_cat.jpg) |

## Summary

To set up the inference process using the pre-trained MaskRCNN was relatively easy. The model author provides image preprocessor and postprocessor functions. To make it work in the notebook, we made a few modifications. The predictions of segments, masks, and classes are good especially if we have relatively well-separated objects. For more complex images it can observe overlapping of bounding boxes and masks between different objects.
