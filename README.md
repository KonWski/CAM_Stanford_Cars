# CAM_STANFORD_CARS
This repository is an implementation of AlexNet generating Class Activation Maps (CAM). The whole model was written in Python using PyTorch library. Data used for training comes from Stanford Cars dataset available through Torchvision.

# CAM
Class activation mapping is a special procedure mounted on top of traditional convolutional neural network which outputs a heatmap showing which pixels of image were significant during prediction. It was primarly introduced in [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150) paper by Zhou et al.

| ![cam_structure](/images/cam_structure.png) |
|:--:|
| Class Activation Mapping[1]|

As presented in above picture class activation map of class $c$ is a weighted sum of $n$ channels taken from last convolutional layer and weights attached to the output neuron representing class $c$.

Assuming that the last convolutional layer is made up of $n$ units, let $f_k(x,y)$ be an activation of unit $k$ ($k \leq n$) at coordinates $(x,y)$. Let's also assume that $w^c_k$ is $k$'th weight of neuron representing class $c$

Therefore class activation map for each of the coordinates for class $c$ has the following formula:

$M_c(x,y) = \sum_{k} w^c_k f_k(x,y)$ 

[1]: https://arxiv.org/pdf/1512.04150.pdf

# How to work with project

## Training a model
```
!python /CAM_Stanford_Cars/main.py --n_epochs 30 \
                                   --batch_size 4 \
                                   --checkpoints_dir 'path/to/dir/for/checkpoints'\
                                   --download_datasets 'true'\
                                   --root_datasets_dir 'stanford_cars_dataset' \
                                   --car_type 'Sedan' \
                                   --car_brand 'BMW' \
                                   --car_production_year 2012
```
Args used in command:
- n_epochs - number of epochs
- batch_size - number of images in batch
- checkpoint_dir - path to directory where checkpoint will be saved
- download_datasets - download dataset from Torchvision repo or use already existing dataset
- root_datasets_dir - path where dataset should be downloaded or where is it already stored
- car_type - limit records by given car_type
- car_brand - limit records by given car_brand
- car_production_year - limit records by given year of production

## Visualizing cams
Import libraries:
```python
from torch.utils.data import DataLoader
from CAM_Stanford_Cars.dataset import StanfordCarsCAM
from CAM_Stanford_Cars.model import load_checkpoint
from CAM_Stanford_Cars.visualize import apply_cam, visualize_cam
```

Load checkpoint:
```python
checkpoint_path = "path/to/Your/checkpoint"
model, checkpoint = load_checkpoint(checkpoint_path)
```

Initialise Dataset and Dataloader (works fine for batch_size > 1):
```python
cars_dataset = StanfordCarsCAM("stanford_cars_dataset", 
                               car_type = checkpoint["car_type"], 
                               car_production_year = checkpoint["car_production_year"], 
                               car_brand = checkpoint["car_brand"], 
                               download_datasets=True, 
                               generate_img_for_cam=True,
                               split="test")

cars_dataloader = DataLoader(cars_dataset, batch_size=1, shuffle=True)
```

Actual visualization:
```python
for id, batch in enumerate(cars_dataloader):

    # extract original image, transformed image and labels
    image_predict, image_visualize, labels = batch

    # generate classification output and cam
    output, cam = model(image_visualize, True)

    # decode output
    predicted_classes = [cars_dataset.class_to_idx[element] for element in torch.argmax(output, 1).tolist()]

    # detach cam for visualization purpose
    cam = cam.detach()
    
    # apply cam on image and visualize it
    blended_images = apply_cam(image_visualize, cam)
    visualize_cam(image_visualize, blended_images, predicted_classes)
    
    break
```
Output:

![cam_example](/images/cam_example.png)