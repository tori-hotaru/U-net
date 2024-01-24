import os
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor() # 意思是将PIL.Image或者numpy.ndarray转化为tensor，归一化至[0, 1]
])

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
    
    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, index):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)


if __name__ == '__main__':
    data = MyDataset('/Users/sunshizhe/Downloads/VOCdevkit/VOC2012')
    print(data[0][0].shape) # 为什么是data[0][0]而不是data[0]呢？因为data[0]是一个tuple，tuple的第一个元素是image，第二个元素是segment_image
    print(data[0][1].shape)
