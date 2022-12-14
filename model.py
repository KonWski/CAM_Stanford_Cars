from torchvision.models import alexnet
import torch
from torch import nn
from torch.nn.functional import softmax
from torch import Tensor
import logging

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
        predicted_classes = torch.argmax(output, 1)
        cams = []

        for n_image, predicted_class in enumerate(predicted_classes.tolist()):
            
            # select weights for predicted class
            weights = self.classifier.weight[predicted_class]
            weights = weights.reshape(256, 1)

            # select features from batch
            features_image = features[n_image]
            features_image = features_image.reshape(256, 49)

            # sum of conv layers multiplied by weights
            weights_softmax = softmax(weights)
            cam = features_image.mul(weights_softmax)

            # sum all elements across channel per batch element
            cam = cam.sum(0)

            # normalize to 0-1
            max = torch.max(cam)
            min = torch.min(cam)
            cam = (cam - min) / (max - min)

            # reshape to the original size
            cam = cam.reshape(1, 1, 7, 7)
            cam = torch.nn.functional.interpolate(cam, (256, 256), mode='bilinear')

            cams.append(cam)

        # convert list into batch of cams
        cams = torch.concat(cams)

        return cams


def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    '''
    saves checkpoint on given checkpoint_path
    '''
    torch.save(checkpoint, checkpoint_path)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(8*"-")


def load_checkpoint(checkpoint_path: str):
    '''
    loads model checkpoint from given path

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint

    Notes
    -----
    checkpoint: dict
                parameters retrieved from training process i.e.:
                - model_state_dict
                - last finished number of epoch
                - save time
                - params for Stanford Cars dataset 
                    + car type
                    + car brand
                    + car production year
                - number of classes
                - loss from last epoch testing
                - accuracy from last epoch testing
                
    '''
    checkpoint = torch.load(checkpoint_path)
    n_classes = checkpoint["n_classes"]

    # initiate model
    model = AlexnetCam(n_classes)

    # load parameters from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])

    # print loaded parameters
    logging.info(f"Loaded model from checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(f"Save dttm: {checkpoint['save_dttm']}")
    logging.info(f"Car type: {checkpoint['car_type']}")
    logging.info(f"Car brand: {checkpoint['car_brand']}")
    logging.info(f"Car production year: {checkpoint['car_production_year']}")
    logging.info(f"Number of classes: {checkpoint['n_classes']}")
    logging.info(f"Test loss: {checkpoint['test_loss']}")
    logging.info(f"Test accuracy: {checkpoint['test_accuracy']}")

    logging.info(8*"-")

    return model, checkpoint