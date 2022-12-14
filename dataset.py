from PIL import Image
import torch.utils.data as data
from cars_transforms import transform_visualize, transform_predict
from torchvision.datasets import StanfordCars
import pandas as pd
from torchvision import transforms

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
        limits records by given car_brand
    car_type: str
        limits records by given car_type
    car_production_year: int
        limits records by given car_production_year
    
    '''
    def __init__(
        self, 
        root: str, 
        split: str, 
        car_brand: str = None, 
        car_type: str = None,
        car_production_year: int = None,
        download_datasets: bool = False,
        generate_img_for_cam: bool = False,
        transform_prediction: transforms.Compose = transform_predict, 
        transform_visualization: transforms.Compose = transform_visualize,
    ) -> None:
        
        super().__init__(root=root, split=split, download=download_datasets)
        self.car_brand = car_brand
        self.car_type = car_type
        self.car_production_year = car_production_year
        self.generate_img_for_cam = generate_img_for_cam
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
        if self.generate_img_for_cam:

            image_visualize = self.transform_visualization(image)
            return image_predict, image_visualize, target

        return image_predict, target


    def _filter_samples(self):

        updated_samples_list = []
        old_idxs = set(self.classes_specification["old_idx"].to_list())
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
        car_production_years = []

        for car_class in self.class_to_idx.keys():
            
            car_classes.append(car_class)
            old_idxs.append(self.class_to_idx[car_class])
            class_record = car_class.split(" ")
            car_brands.append(class_record[0] if class_record[0] != "Land" else "Land Rover")
            car_production_years.append(int(class_record[-1]))
            car_types.append(class_record[-2])

            specification = pd.DataFrame({
                "car_class": car_classes,
                "old_idx": old_idxs,
                "car_brand": car_brands,
                "car_type": car_types,
                "car_production_year": car_production_years
                })

        # filter specification by car_brand
        if self.car_brand:
            specification = specification[specification["car_brand"].isin(self.car_brand.split())]
        
        # filter specification by car_type
        if self.car_type:
            specification = specification[specification["car_type"].isin(self.car_type.split())]

        # filter specification by car_production_year
        if self.car_production_year:
            specification = specification[specification["car_production_year"] == self.car_production_year]

        # adjust ids for model
        specification["new_idx"] = range(len(specification))

        return specification