import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    tracking_dir = Path(args.tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    experiment_name = f"Assignment3_{args.name}"
    mlflow.set_experiment(experiment_name)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    data_root = Path(args.data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    full_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    with mlflow.start_run():
        mlflow.set_tag("student_id", args.student_id)
        mlflow.set_tag("course", "Assignment3 Observable ML")

        mlflow.log_params(
            {
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": "SGD",
                "seed": args.seed,
                "model": "SimpleMLP",
                "dataset": "MNIST",
            }
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total
            val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)

          
            mlflow.log_metric("loss", train_loss, step=epoch)
            mlflow.log_metric("accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        mlflow.pytorch.log_model(model, artifact_path="model")
        print(f"Final test_loss={test_loss:.4f} test_accuracy={test_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLflow")
    parser.add_argument("learning_rate", type=float, default=0.01)
    parser.add_argument("epochs", type=int, default=3)
    parser.add_argument("batch_size", type=int, default=64)
    parser.add_argument("seed", type=int, default=42)
    parser.add_argument("name", type=str, default="Joumana")
    parser.add_argument("student_id", type=str, default="202202100")
    parser.add_argument("tracking_dir", type=str, default="./mlruns")
    parser.add_argument("data_dir", type=str, default="./data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
