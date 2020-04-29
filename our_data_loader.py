import os
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
from skimage.io import imread
from PIL import Image

from common_flags import COMMON_FLAGS

np.random.seed(5)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        super(Dataset, self).__init__()
        self.transform = transform
        self.img_dir = os.path.join(COMMON_FLAGS.dataset_dir)
        self.labels = np.loadtxt(os.path.join(COMMON_FLAGS.root_dir, 'dataset', 'labels.txt'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error:' + str(index)+'.jpg')
            item = self.load_item(0)

        return item
    
    def load_item(self, index):
        img = imread(os.path.join(self.img_dir, str(index)+'.jpg')) # [0,255] uint8.
        # if len(img.shape) == 2:
        #     img = np.concatenate([img, img, img], axis=-1)
        assert len(img.shape) == 3
        # print('img:', type(img), img.shape)
        img = Image.fromarray(img)
        # print('img:', type(img), img.size)

        img = self.transform(img)
        # print('img:', type(img), img.size())

        label = int(self.labels[index])

        return img, label, index # img_file_name is str(index) + '.jpg'

def dataloader(batch_size, img_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.Resize(int(img_size/0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = Dataset(transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return test_loader

def my_effnet_loader(batch_size):
    import PIL
    from efficientnet_pytorch import EfficientNet
    image_size = EfficientNet.get_image_size('efficientnet-b7')
    print('image_size:', image_size)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
           transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
           transforms.CenterCrop(image_size),
           transforms.ToTensor(),
           normalize,
       ])
    test_dataset = Dataset(transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return test_loader


if __name__ == '__main__':
    test_loader = dataloader(50)
    print('test_loader:', len(test_loader))
    

