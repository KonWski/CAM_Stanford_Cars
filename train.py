import logging
from torch.optim import Adam
import torch
from torch import nn
from torch.nn.functional import softmax
from torchvision import datasets, models
from transforms import transform_predict
from model import AlexnetCam

def train_model(
        device, 
        n_epochs: int,
        batch_size: int,
        checkpoints_dir: str,
        download_datasets: bool,
        root_datasets_dir: str
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
    '''

    model = AlexnetCam()
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)

    # datasets and dataloaders
    trainset = datasets.StanfordCars(f'{root_datasets_dir}/train/', split="train", download=download_datasets, transform=transform_predict)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.StanfordCars(f'{root_datasets_dir}/test/', split="test", download=download_datasets, transform=transform_predict)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # number of observations
    len_train_dataset = len(trainset)
    len_test_dataset = len(testset)

    for epoch in range(n_epochs):

        for state, loader, len_dataset in zip(["train", "test"], [train_loader, test_loader], [len_train_dataset, len_test_dataset]):

            logging.info(f"Epoch {epoch}")
                
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
                    # print(f"labels.shape: {labels.shape}")
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    # calculate loss
                    outputs = model(images).to(device)
                    # print(f"outputs.shape: {outputs.shape}")
                    loss = criterion(outputs, labels)

                    proba = softmax(outputs)
                    # print(f"proba.shape: {proba.shape}")
                    preds = torch.round(proba)
                    # print(f"preds.shape: {preds.shape}")

                    if state == "train":
                        loss.backward()
                        optimizer.step()

                # print statistics
                running_loss += loss.item()
                # print(f"torch.argmax(preds): {torch.argmax(preds)}")
                # print(f"torch.argmax(labels): {torch.argmax(labels)}")
                running_corrects += (torch.argmax(preds, dim=1) == torch.argmax(labels, dim=1)).sum().item()

            # save and print epoch statistics
            epoch_loss = round(running_loss / len_dataset, 2)
            epoch_acc = round(running_corrects / len_dataset, 2)

            logging.info(f"Epoch: {epoch}, loss: {epoch_loss}, accuracy: {epoch_acc}")

        # save model to checkpoint
        checkpoint = {"epoch": epoch, 
                      "loss": epoch_loss, 
                      "accuracy": epoch_acc, 
                      "model_state_dict": model.state_dict(), 
                      "optimizer_state_dict": optimizer.state_dict()}

        checkpoint_path = f"{checkpoints_dir}/AlexnetCam_{epoch}"
        save_checkpoint(checkpoint, checkpoint_path)

    return model


def save_checkpoint(checkpoint: dict, checkpoint_path: str):

    torch.save(checkpoint, checkpoint_path)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(8*"-")