from PIL import Image
import torch.utils.data as data
from transforms import tranform_visualize, transform_predict
from torchvision.datasets import StanfordCars
import scipy.io as sio
import pandas as pd
from typing import Callable, Optional
from typing import Callable, Optional


class StanfordCarsCAM(StanfordCars):
    '''
    Dataset bounding Stanford dataset to given category:
    brand, car_type

    Full path to dataset is defined by:
    {self.root_dir}/{self.state}/stanford_cars/cars_{self.state}

    Attributes
    ----------
    root_dir: str
        path to stanford_cars_dataset content
    state: str
        train or test
    car_brand: str
        examples: 
        all -> takes all brands
        bwm -> select only bmw
    car_type: str
        for example coupe will select only coupe types of cars
    '''
    def __init__(
        self, 
        root: str, 
        split: str, 
        car_brand: str = None, 
        car_type: str = None,
        download_dataset: bool = False,
        transform_prediction: Optional[Callable] = None, 
        transform_visualization: Optional[Callable] = None     
    ) -> None:
        
        super().__init__(root=root, split=split, download=download_dataset)
        self.car_brand = car_brand
        self.car_type = car_type
        self.transform_prediction = transform_prediction
        self.transform_visualization = transform_visualization
        self.classes_specification = self._classes_specification()
        
        # update inherited fields according to filter conditions
        self.classes = self.classes_specification["car_class"].to_list()
        self.class_to_idx = {idx: car_class for idx, car_class in 
                                zip(self.classes_specification["new_idx"].to_list(), 
                                    self.classes_specification["car_class"].to_list())}
        self._samples = self._filter_samples()


    def __getitem__(self, idx):
        
        image_path, target = self._samples[idx]
        image = Image.open(image_path).convert("RGB")

        image_predict = self.transform_prediction(image)

        # generate transformed image for cam purpose
        if self.transform_visualization:

            image_visualize = self.transform_visualization(image)
            return image_predict, image_visualize, target

        return image_predict, target


    def _filter_samples(self):

        updated_samples_list = []
        old_idxs = set(self.specification["old_idxs"].to_list())
        old_to_new_idxs = {idx: car_class for idx, car_class in 
                                zip(self.classes_specification["old_idx"].to_list(), 
                                    self.classes_specification["new_idx"].to_list())}

        for sample in self._samples:
            path = sample[0]
            old_idx = sample[1]

            if old_idx in old_idxs:
                new_idx = old_to_new_idxs[old_idx]
                updated_samples_list.append((path, new_idx))

        return updated_samples_list


    def _classes_specification(self):
        '''
        creates specification of samples including:
        - full class name
        - class id
        - brand
        - type
        - year of production
        '''        
        car_classes = []
        old_idxs = []
        car_brands = []
        car_types = []
        car_years = []

        for car_class in self.class_to_idx.keys():
    
            car_classes.append(car_class)
            old_idxs.append(self.class_to_idx[car_class])
            class_record = car_class.split(" ")
            car_brands.append(class_record[0] if class_record[0] != "Land" else "Land Rover")
            car_years.append(class_record[-1])
            car_types.append(class_record[-2])

            specification = pd.DataFrame({
                "car_class": car_classes,
                "old_idx": old_idxs,
                "car_brand": car_brands,
                "car_year": car_years,
                "car_type": car_types}
                )
        
        # filter specification by car_brand
        if self.car_brand:
            specification = specification[specification["car_brand"] == self.car_brand]
        
        # filter specification by car_type
        if self.car_type:
            specification = specification[specification["car_type"] == self.car_type]

        # adjust ids for model
        specification["new_idx"] = range(len(specification))

        return specification


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