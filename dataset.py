from PIL import Image
import torch.utils.data as data
from transforms import tranform_visualize, transform_predict

class StanfordCarsVisualizeCAM(data.Dataset):
    '''
    Dataset enabling handling Stanford cars dataset in such way
    that it returns two transformed images - for prediction and for visualization.

    Full path to dataset is defined by:
    {self.root_dir}/{self.state}/stanford_cars/cars_{self.state}

    Attributes
    ----------
    root_dir: str
        path to stanford_cars_dataset content
    state: str
        train or test
    '''

    def __init__(self, root_dir: str, state: str):
        super().__init__()
        self.root_dir = root_dir
        self.state = state
        self.transform_predict = transform_predict
        self.tranform_visualize = tranform_visualize

    def __getitem__(self, index):
        
        image_id = "0" * (5 - str(index))
        image_path = f"{self.root_dir}/{self.state}/stanford_cars/cars_{self.state}/{image_id}.jpg"
        image = Image.open(image_path)

        image_predict = self.transform_prediction(image)
        image_visualize = self.tranform_visualization(image)

        return image_predict, image_visualize