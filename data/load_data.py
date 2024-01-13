import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

# from modelconfig import config as mcfg


def load():
    # 对图像做变换
    # data_transform = transforms.Compose(
    #     [
    #         transforms.Resize(1200),
    #         transforms.RandomCrop((800,800)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # # 数据集
    # # 训练集 其中 0：明  1：清  2：民国
    # full_dataset=ImageFolder(root="/root/autodl-tmp/data/", transform=data_transform)
    # #85%的图片是训练集
    # train_size = int(0.85 * len(full_dataset))
    # test_size = len(full_dataset) - train_size

    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # print(len(train_dataset),len(test_dataset))

    # train_loader = DataLoader(train_dataset, batch_size=mcfg.batch_size, shuffle=True, num_workers=12)
    # test_loader = DataLoader(test_dataset, batch_size=mcfg.batch_size, shuffle=True, num_workers=12)

    # return (train_loader, test_loader)

    training_data = datasets.CIFAR10(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=ToTensor()
    )
    train_loader = DataLoader(training_data, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=2)

    return (train_loader, test_loader)


if __name__ == "__main__":
    train_loader, test_loader = load()
    train_features, train_labels = next(iter(train_loader))
    label = train_labels[0]
    # classes=("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    print(f"Label: {label}")
    print(label)
