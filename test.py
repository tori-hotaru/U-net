from net import *
import os
from utils import *
from data import *
from torchvision.utils import save_image

net = UNet()

weights = './params/unet.pth'

if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('load weights success!')
else:
    print('load weights failed!')


_input = input('input image path:')

img = keep_image_size_open(_input)
img_data = transform(img)
print(img_data.shape)
img_data = torch.unsqueeze(img_data, dim=0) # unsqueeze的意思是在img_data的第0个维度上增加一个维度，因为net的输入是4维的，而img_data是3维的
out = net(img_data)
print(out)
save_image(out, './result_image/test.png')
