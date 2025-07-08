import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

def main():
    transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net=LeNet()
    net.load_state_dict(torch.load('lenet.pth'))

    im=Image.open('test.jpg')
    im=transform(im)
    im=transform(im).unsqueeze(im,dim=0)

    with torch.no_grad():
        output=net(im)
        predict=torch.max(output,dim=1)[1].numpy()
    print(classes[int(predict)])

if __name__ == '__main__':
    main()