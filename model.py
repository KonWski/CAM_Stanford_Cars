from torchvision.models import alexnet
import torch
from torch import nn
from torch.nn.functional import softmax
from torch import Tensor

class AlexnetCam(nn.Module):
    '''
    Class activation map model based on pretrained AlexNet.    

    Attributes
    ----------
    features: nn.Sequantial
        features part of Torchvision Alexnet model
    gap: nn.AvgPool2d
        global average pooling: (256, 7, 7) -> (256, 1, 1)
    classifier: nn.Linear
        last, fully connected layer
    n_classes: int
        number of classes - number of output neurons
    '''

    def __init__(self, n_classes: int):

        super().__init__()

        self.features = nn.Sequential(*list(alexnet(pretrained=True).children())[:-2])
        self.gap = nn.AvgPool2d(7)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x: Tensor, get_cam: bool = False):
        '''
        Runs forward pass with x tensor and generates cam if get_cam
        set on true.

        Parameters
        ----------
        x: Tensor
            tensor representing image
        get_cam: bool = False
            parameter set on True generates cam for the picture
        '''
        
        features = self.features(x)
        gap = self.gap(features)
        gap_flattened = torch.flatten(gap, 1)
        output = self.classifier(gap_flattened)

        # class activation map
        if get_cam:
            cam = self._generate_cam(features, output)
            return output, cam

        return output
    
    def _generate_cam(self, features: Tensor, output: Tensor):
        '''
        Generates cam by multiplying connection weights of
        the neuron, that has the highest score, with corresponding channel
        of last layer in features part of the model.

        Parameters
        ----------
        features: Tensor
            tensor representing features generated by model
        output: Tensor
            tensor representing scores of output neurons
        '''
        batch_size = features.shape[0]

        predicted_class = torch.argmax(output)

        weights = self.classifier.weight[predicted_class]
        weights = weights.reshape(256, 1)
        weights = weights.repeat(batch_size, 1, 1)

        features = features.reshape(batch_size, 256, 49)

        # sum of conv layers multiplied by weights
        weights_softmax = softmax(weights)
        cam = features.mul(weights_softmax)
        cam = cam.reshape(batch_size, 256, 7, 7)
        
        # sum all elements across channel per batch element
        cam = cam.sum(1)

        # normalize to 0-1
        max = torch.max(cam)
        min = torch.min(cam)
        cam = (cam - min) / (max - min)

        # reshape to the original size
        cam = cam.reshape(batch_size, 1, 7, 7)
        cam = torch.nn.functional.interpolate(cam, (256, 256), mode='bilinear')

        return cam