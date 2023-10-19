import argparse
import os
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models


def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'valid']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                   for x in ['train', 'valid']}

    return dataloaders, image_datasets


def build_model(arch, hidden_units):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model


def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, use_gpu):
    dataloaders, _ = load_data(data_dir)
    model = build_model(arch, hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(dataloaders['train']):.4f}")

    model.class_to_idx = dataloaders['train'].dataset.class_to_idx

    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,  # Save the classifier
    }

    os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a deep learning model on a dataset.")
    parser.add_argument('data_dir', type=str, help="Path to the data directory")
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help="Directory to save checkpoints")
    parser.add_argument('--arch', type=str, default='vgg16', help="Choose architecture (e.g., 'vgg16')")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--hidden_units', type=int, default=256, help="Number of hidden units in the classifier")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training")

    args = parser.parse_args()
    train_model(data_dir=args.data_dir, save_dir=args.save_dir, arch=args.arch, learning_rate=args.learning_rate,
                hidden_units=args.hidden_units, epochs=args.epochs, use_gpu=args.gpu)