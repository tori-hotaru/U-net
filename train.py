import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = './params/unet.pth'
data_path = r'/Users/sunshizhe/Downloads/VOCdevkit/VOC2012'
save_path = './train_image'

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=2, shuffle=True) # batch_sizey意味着每次训练的时候，训练的图片数量是2张
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('success load weight')
    else:
        print('weight not exist')
    

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    epoch = 1 
    while True:
        for i, (image, segment_image) in enumerate(data_loader):
            image = image.to(device)
            segment_image = segment_image.to(device)
            out_image = net(image) # 显示的是网络的输出
            train_loss = loss_fun(out_image, segment_image)
            opt.zero_grad() # 梯度清零，否则会累加
            train_loss.backward()
            opt.step()
            if i % 5 == 0:
                print('epoch: {}, step: {}, loss: {}'.format(epoch, i, train_loss.item()))
            if i % 50 == 0:
                torch.save(net.state_dict(), weight_path)
            
            _image = image[0] # 为什么是image[0]而不是image呢？因为image是一个tensor，tensor的第一个元素才是imag，第二个元素是segment_image
            _segment_image = segment_image[0]
            _out_image = out_image[0] 
            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png') # 意思是将img保存到save_path下，命名为i.png
            
        epoch += 1


