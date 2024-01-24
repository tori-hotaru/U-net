from PIL import Image

def keep_image_size_open(path, size=(256,256)): #size是一个tuple，(256,256)表示将图片resize成256*256的大小
    image = Image.open(path)
    temp = max(image.size)
    mask = Image.new('RGB', (temp, temp), (0,0,0))
    mask.paste(image, (0,0))
    mask = mask.resize(size)
    return mask