# Experiment with Single Shot MultiBox Detector (SSD)

To run experiments with the [SSD](https://arxiv.org/abs/1512.02325) model use [onnx_ssd.ipynb](../onnx_ssdn.ipynb) notebook. Using this notebook we conducted experiments with the SSD deep neural network model aimed to solve objects detection problem in computer vision.

## Dataset

For prediction experiments, we are using an "animal" data set. This dataset combines photographs cat(s) and dog(s) collected from the internet.

| Animal | Original Image | Preprocessed Image | Postprocessed Image | Score Threshold |
| -------| -------------- |------------------- | ------------------- | --------------- |
| cat-1| ![orig_cat_1](images/orig_cat_1.jpg) | ![ssd_pre_cat_1](images/ssd_pre_cat_1.jpg) | ![ssd_post_cat_1](images/ssd_post_cat_1.jpg)| 0.7 |
| dog-1| ![orig_dog_1](images/orig_dog_1.jpg) | ![ssd_pre_dog_1](images/ssd_pre_dog_1.jpg) | ![ssd_post_dog_1](images/ssd_post_dog_1.jpg)| 0.7 |
| cat-n| ![orig_cat_n](images/orig_cat_n.jpg) | ![ssd_pre_cat_n](images/ssd_pre_cat_n.jpg) | ![ssd_post_cat_n](images/ssd_post_cat_n.jpg)| 0.15 |
| dog-n| ![orig_dog_n](images/orig_dog_n.jpg) | ![ssd_pre_dog_n](images/ssd_pre_dog_n.jpg) | ![ssd_post_dog_n](images/ssd_post_dog_n.jpg)| 0.2 |
| dog-n-cat | ![orig_dog_n_cat](images/orig_dog_n_cat.jpg) | ![ssd_pre_dog_n_cat](images/ssd_pre_dog_n_cat.jpg) | ![ssd_post_dog_n_cat](images/ssd_post_dog_n_cat.jpg) | 0.3 |