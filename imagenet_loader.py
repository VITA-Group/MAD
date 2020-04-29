import torch
from torchvision import datasets, transforms

def imagenet_loader(imagenet_dir, train_batch_size, val_batch_size,  
                    img_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    imagenet dataloader.
    after crop: 224, 299, 331
    before crop: 256, 342, 378
    '''
    print('imagenet_dir:', imagenet_dir)
    print('img_size:', img_size)
    print('mean:', mean)
    print('std:', std)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=mean, std=std)

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    test_transform = transforms.Compose([
        transforms.Resize(int(img_size/0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageNet(imagenet_dir, split='train', download=False, transform=train_transform),
    #     batch_size=train_batch_size, shuffle=True, pin_memory=True)
    train_loader = None

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageNet(imagenet_dir, split='val', download=True, transform=test_transform),
        batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, val_loader

def effnet_loader(imagenet_dir, batch_size):
    import PIL
    from efficientnet_pytorch import EfficientNet
    image_size = EfficientNet.get_image_size('efficientnet-b7')
    print('image_size:', image_size)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
           transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
           transforms.CenterCrop(image_size),
           transforms.ToTensor(),
           normalize,
       ])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageNet(imagenet_dir, split='val', download=True, transform=val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return val_loader
