import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def apply_cam(images: torch.Tensor, cam: torch.Tensor):
    '''
    applies cam on given images

    Parameters
    ----------
    images: torch.Tensor
        batch of images transformed by visualization transormations
    cam: torch.Tensor
        model's output (when images are input)
    '''

    to_tensor_transform = transforms.ToTensor()

    # batch_size heatmaps - cams
    batch_size = cam.shape[0]
    cam_heatmaps = torch.cat((cam, torch.zeros(batch_size, 2, 256, 256)), dim=1)

    # blend each original picture with cam
    blended_images = []

    for n_image in range(batch_size):

        cam_heatmap = TF.to_pil_image(cam_heatmaps[n_image])
        image = TF.to_pil_image(images[n_image])

        # add blended image to returned list
        blended_image = Image.blend(image, cam_heatmap, 0.5)
        blended_image = to_tensor_transform(blended_image)
        
        blended_image = blended_image.permute(1, 2, 0).detach().numpy()
        blended_images.append(blended_image)

    return blended_images


def visualize_cam(image: torch.Tensor, blended_image: np.ndarray):
    '''
    shows pair of images - before and after applying cam
    
    Parameters
    ----------
    image: torch.Tensor
        single image before applying cam
    blended_image: np.ndarray
        original image blended with heatmap
    '''
    fig, ax = plt.subplots(1, 2)
    ax[0, 0].imshow(image)
    ax[0, 1].imshow(blended_image[:, :, 0], cmap="jet")