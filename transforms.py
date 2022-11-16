from torchvision import transforms

transform_predict = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomGrayscale(0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(hue=0.3),
            transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

tranform_visualize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])