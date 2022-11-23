import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image


def apply_cam(images: torch.Tensor, cam: torch.Tensor):
    '''
    applies cam on given images

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

        # single cam, image
        cam_heatmap = TF.to_pil_image(cam_heatmaps[n_image])
        image = TF.to_pil_image(images[n_image])

        # add blended image to returned list
        blended_image = Image.blend(image, cam_heatmap, 0.5)
        blended_image = to_tensor_transform(blended_image)
        
        blended_image = blended_image.permute(1, 2, 0).detach().numpy()
        blended_images.append(blended_image)

    return blended_images