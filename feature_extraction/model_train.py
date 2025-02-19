import os
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets as dset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# データセットの準備関数
def prepare_dataset(dataroot, image_size, subset_size=None):
    dataset = dset.ImageFolder(
        root=dataroot,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    if subset_size is not None:
        total_size = len(dataset)
        indices = random.sample(range(total_size), subset_size)
        dataset = Subset(dataset, indices)
    return dataset

# DataLoaderの作成関数
def create_dataloader(dataset, batch_size, workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# モデルの準備関数
def prepare_model(num_classes):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 学習関数
def train_model(model, dataloader, device, num_epochs, lr, beta1, outputdir):
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    
    print("Starting Training Loop...")
    now = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
    
    for epoch in range(num_epochs):
        epoch_corrects = 0
        
        for i, data in enumerate(tqdm(dataloader), 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            
            epoch_corrects += torch.sum(preds == labels)
        
        epoch_acc = epoch_corrects.double() / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Accuracy: {epoch_acc:.4f}")
    
    os.makedirs(outputdir, exist_ok=True)
    save_path = os.path.join(outputdir, f"{now}_resnet50_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# 評価モードで画像表示関数
def evaluate_model(model, dataloader, device):
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data[0].to(device)
            
            if i == 0:
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title("Sample Training Images")
                plt.imshow(
                    np.transpose(
                        torchvision.utils.make_grid(inputs[:64], padding=2, normalize=True).cpu(),
                        (1, 2, 0)
                    )
                )
                plt.show()
                break

# メイン関数（全体の実行フロー）
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset = prepare_dataset(dataroot=args.dataroot,
                              image_size=args.image_size,
                              subset_size=args.subset_size)
    
    dataloader = create_dataloader(dataset=dataset,
                                   batch_size=args.batch_size,
                                   workers=args.workers)
    
    model = prepare_model(num_classes=args.num_classes)
    
    train_model(model=model,
                dataloader=dataloader,
                device=device,
                num_epochs=args.num_epochs,
                lr=args.lr,
                beta1=args.beta1,
                outputdir=args.outputdir)

    evaluate_model(model=model,
                   dataloader=dataloader,
                   device=device)

# コマンドライン引数の設定
# ex) python model_train.py --dataroot '/path/to/dataset' --outputdir '/path/to/output' --image_size 128 --subset_size 1000 --batch_size 32 --num_classes 6 --num_epochs 10 --lr 0.001 --beta1 0.9

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a feature extraction model on a custom dataset.")
    
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to the dataset root directory.')
    parser.add_argument('--outputdir', type=str, required=True,
                        help='Directory to save the trained model weights.')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Input image size (default: 128).')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Number of images to use from the dataset (default: use all).')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32).')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of worker threads for DataLoader (default: 2).')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of output classes (default: 6).')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs (default: 10).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001).')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 hyperparameter for Adam optimizer (default: 0.9).')
    
    args = parser.parse_args()
    
    main(args)
