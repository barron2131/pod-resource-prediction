import os
import torch
from models import GATGRU
from data import PodDataset
from torch.utils.data import DataLoader, random_split
from argparse import ArgumentParser
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for (X, Ax), (y, Ay) in dataloader:
        X, Ax, y, Ay = X.to(device), Ax.to(device), y.to(device), Ay.to(device)
        p = model(X, Ax)
        loss = loss_fn(p, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for (X, Ax), (y, Ay) in dataloader:
            X, Ax, y, Ay = X.to(device), Ax.to(device), y.to(device), Ay.to(device)
            p = model(X, Ax)
            loss = loss_fn(p, y)
            test_loss += loss.item()

    return test_loss / len(dataloader)


def plot_train_test_curves(train_history, val_history):
  plt.title(f'Loss curves')
  plt.plot(train_history, label='Train')
  plt.plot(val_history, label='Validation')
  plt.xlabel('Epoch')
  plt.ylabel('Loss (MSE)')
  plt.legend()
  plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description="GAT-GRU model trainer")
    parser.add_argument("--data", required=True, type=str,
                        help="Datasource directory for training")
    parser.add_argument("--batch-size", default=8, type=int,
                        help="Batch size for training")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of training epochs")
    parser.add_argument("--early-stopping-patience", default=None, type=int,
                        help="Early stopping patience for")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--lr-decay", default=1e-3, type=float, help="Learning rate decay")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    parser.add_argument("--heads", default=2, type=int, help="Number of attention heads")
    parser.add_argument("--output-model", default=None, type=str, help="Output file for trained model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = PodDataset(args.data)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    model = GATGRU(
        in_features=2, hidden_channels=5, num_nodes=11,
        window_len=60, dropout_p=args.dropout).to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)

    train_history = []
    val_history = []
    min_loss = float("inf")
    early_epochs = 0

    with TemporaryDirectory() as tmp_dir:
        weights_path = os.path.join(tmp_dir, "best_weights")

        for t in tqdm(range(args.epochs)):
            train_loss = train(model, train_dataloader, loss_fn, optimizer, device)
            val_loss = evaluate(model, val_dataloader, loss_fn, device)
            train_history.append(train_loss)
            val_history.append(val_loss)

            if val_loss > min_loss:
                early_epochs += 1
                if early_epochs == args.early_stopping_patience:
                    model.load_state_dict(torch.load(weights_path))
                    break
            else:
                min_loss = val_loss
                early_epochs = 0
                torch.save(model.state_dict(), weights_path)
    if args.output_model is not None:
        torch.save(model.state_dict(), args.output_model)

    plot_train_test_curves(train_history, val_history)
