import argparse
import logging
import torch
from train import train_model

def get_args():

    parser = argparse.ArgumentParser(description='Paramaters for model training')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Number of images in batch')
    parser.add_argument('--checkpoints_dir', type=str, help='Path to directory where checkpoint will be saved')
    parser.add_argument('--download_datasets', type=str, help='Download dataset from Torchvision repo or use already existing dataset')
    parser.add_argument('--root_datasets_dir', type=str, help='Path where dataset should be downloaded or where is it already stored')
    parser.add_argument('--car_type', type=str, help='Limit records by given car_type')
    parser.add_argument('--car_brand', type=str, help='Limit records by given car_brand')
    parser.add_argument('--car_production_year', type=int, help='Limit records by given year of production') 

    args = vars(parser.parse_args())
    
    # parse str to boolean
    str_true = ["Y", "y", "Yes", "yes", "true", "True"]
    bool_params = ["download_datasets"]
    for param in bool_params:
        if args[param] in str_true:
            args[param] = True
        else:
            args[param] = False

    # log input parameters
    logging.info(8*"-")
    logging.info("PARAMETERS")
    logging.info(8*"-")

    for parameter in args.keys():
        logging.info(f"{parameter}: {args[parameter]}")
    logging.info(8*"-")

    return args

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    model = train_model(device, args["n_epochs"], args["batch_size"], args["checkpoints_dir"], 
                        args["download_datasets"], args["root_datasets_dir"], args["car_type"],
                        args["car_brand"], args["car_production_year"])