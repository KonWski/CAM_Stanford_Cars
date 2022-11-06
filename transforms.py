from torchvision import transforms

transform_predict = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

tranform_visualize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])