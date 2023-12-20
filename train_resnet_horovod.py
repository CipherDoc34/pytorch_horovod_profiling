import torch as torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
import argparse
import horovod.torch as hvd
import horovod
import torch.multiprocessing as mp
import torch.utils.data.distributed
# from pathlib import Path
# from tqdm import tqdm
#  salloc --time=2:0:0 --mem=24G --nodes=1 --gpus=4 --account=def-queenspp

def train(image_loader, model, device, optimizer, loss_fn, scheduler, n_epochs):
    model.train()
    model.to(device)
    total_loss = []

    for epoch in range(1, n_epochs + 1):
        print("epoch: ", epoch)
        step_loss = 0
        for image, label in image_loader:
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
    hvd.init()

    parser = argparse.ArgumentParser(description="Train a classifier on ImageNet.")
    parser.add_argument('-e', '--epochs', type=int, default=3, help="Number of epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help="Size of the batch")
    parser.add_argument('-s', '--save_path', type=str, default="model_horovod.pth", help="Save path for the model")
    parser.add_argument('-p', '--loss_graph_path', type=str, default="model_loss_horovod.png", help="Loss graph save location")
    parser.add_argument('-cuda', "--c", type=str, default="Y", help="use cuda")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()

    device = "cpu"
    if args.c == 'y' or args.c == 'Y' and torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(hvd.local_rank())
        print("running on GPU: ", hvd.local_rank())

    torch.set_num_threads(1)

    transform = transforms.Compose([
    #     transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]) 
    
    image_dataset = datasets.CIFAR10("./data", transform=transform, download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        image_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    model = models.resnet50()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)
    loss_fn = torch.nn.CrossEntropyLoss()

    if device == "cuda":
        model.cuda()

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         op=hvd.Adasum)

    total_loss = train(train_loader, model, device, optimizer, loss_fn, scheduler, args.epochs)

    torch.save(model.state_dict(), args.save_path)

    plt.plot(total_loss)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(args.loss_graph_path)
    # plt.show()


if __name__ == "__main__":
    main()