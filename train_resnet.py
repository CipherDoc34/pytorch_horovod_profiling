import torch as torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
import argparse
# from pathlib import Path
from tqdm import tqdm


def train(image_loader, model, device, optimizer, loss_fn, scheduler, n_epochs):
    model.train()
    model.to(device)
    total_loss = []

    for epoch in range(1, n_epochs + 1):
        print("epoch: ", epoch)
        step_loss = 0
        for image, label in tqdm(image_loader, desc="Training", leave=True, ncols=75):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = loss_fn(output, label)

            loss.backward()
            optimizer.step()

            step_loss += loss.item()
        scheduler.step()
    
        total_loss.append(step_loss/len(image_loader))

        print("Loss: ", total_loss[-1])

    return total_loss


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on ImageNet.")
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=4000, help="Size of the batch")
    parser.add_argument('-s', '--save_path', type=str, default="model.pth", help="Save path for the model")
    parser.add_argument('-p', '--loss_graph_path', type=str, default="model_loss.png", help="Loss graph save location")
    parser.add_argument('-cuda', "--c", type=str, default="Y", help="use cuda")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()

    device = "cpu"
    if args.c == 'y' or args.c == 'Y':
        device = "cuda"

    transform = transforms.Compose([
    #     transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]) 

    model = models.resnet50()

    image_dataset = datasets.CIFAR10("./data", transform=transform, download=True)

    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = train(image_loader, model, device, optimizer, loss_fn, scheduler, args.epochs)

    torch.save(model.state_dict(), args.save_path)

    plt.plot(total_loss)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(args.loss_graph_path)
    # plt.show()


if __name__ == "__main__":
    main()