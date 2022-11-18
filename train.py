import logging
from torch.optim import Adam
import torch
from torch import nn
from torch.nn.functional import softmax
from cars_transforms import transform_predict, transform_test
from dataset import StanfordCarsCAM
from model import AlexnetCam

def train_model(
        device, 
        n_epochs: int,
        batch_size: int,
        checkpoints_dir: str,
        download_datasets: bool,
        root_datasets_dir: str,
        car_type: str = None,
        car_brand: str = None,
        car_production_year: int = None
    ):
    '''
    trains AlexnetCam model and saves its checkpoints to location
    given in checkpoints_dir.

    Parameters
    ----------
    device 
    n_epochs: int
        number of training epochs
    batch_size: int
        number of images inside single batch
    checkpoints_dir: str
        path to directory where checkpoints will be stored
    download_datasets: bool
        True -> download dataset from torchvision repo
    root_datasets_dir: str
        path to directory where dataset should be downloaded (download_datasets = True)
        or where dataset is already stored
    car_type: str = None
        limits records by given car_type
    car_brand: str = None
        limits records by given car_brand
    car_production_year: int = None
        limits records by given year of production
    '''

    # datasets and dataloaders
    trainset = StanfordCarsCAM(f'{root_datasets_dir}/train/', split="train", download_datasets=download_datasets, 
        car_type=car_type, car_brand=car_brand, car_production_year=car_production_year)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = StanfordCarsCAM(f'{root_datasets_dir}/test/', split="test", download_datasets=download_datasets, 
        car_type=car_type, car_brand=car_brand, car_production_year=car_production_year)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    logging.info(f"Number of classes: {len(trainset.classes)}")

    # model
    model = AlexnetCam(len(trainset.classes))
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    # number of observations
    len_train_dataset = len(trainset)
    len_test_dataset = len(testset)

    best_test_loss = float("inf")

    for epoch in range(n_epochs):
        
        checkpoint = {}

        for state, loader, len_dataset in zip(["train", "test"], [train_loader, test_loader], [len_train_dataset, len_test_dataset]):

            # calculated parameters
            running_loss = 0.0
            running_corrects = 0

            criterion = nn.CrossEntropyLoss()

            if state == "train":
                model.train()
            else:
                model.eval()

            for id, batch in enumerate(loader, 0):

                with torch.set_grad_enabled(state == 'train'):
                    
                    images, labels = batch

                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    # calculate loss
                    outputs = model(images).to(device)
                    loss = criterion(outputs, labels)

                    proba = softmax(outputs)

                    if state == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += (torch.argmax(proba, dim=1) == labels).sum().item()

            # save and log epoch statistics
            epoch_loss = round(running_loss / len_dataset, 2)
            epoch_acc = round(running_corrects / len_dataset, 2)

            # save stats for potential checkpoint
            checkpoint[f"{state}_loss"] = epoch_loss
            checkpoint[f"{state}_accuracy"] = epoch_acc

            logging.info(f"Epoch: {epoch}, state: {state}, loss: {epoch_loss}, accuracy: {epoch_acc}")

        if checkpoint["test_loss"] < best_test_loss:
            
            # update lowest test loss
            best_test_loss = checkpoint["test_loss"]

            # save model to checkpoint
            checkpoint["epoch"] = epoch
            checkpoint["car_type"] = car_type
            checkpoint["car_brand"] = car_brand
            checkpoint["car_production_year"] = car_production_year
            checkpoint["model_state_dict"] = model.state_dict()
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            checkpoint_path = f"{checkpoints_dir}/AlexnetCam"
            save_checkpoint(checkpoint, checkpoint_path)

        else:
            logging.info(8*"-")
            
    return model


def save_checkpoint(checkpoint: dict, checkpoint_path: str):

    torch.save(checkpoint, checkpoint_path)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(8*"-")