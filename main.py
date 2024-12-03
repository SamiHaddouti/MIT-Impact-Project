import argparse
import datetime
from pathlib import Path
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model.model import SynthesisPredictionModel
from model.dataset import TrainDataset, train_collate_fn
from model.loss import CustomRankLoss
from engine import evaluate_model, train_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser("Set synthesis predictor", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=100, type=int, 
                        help="Number of epochs")
    parser.add_argument("--batch_size", default=16, type=int, 
                        help="Batch size")
    parser.add_argument("--split_ratio", default=0.8, type=float, 
                        help="Train/eval split ratio")
    parser.add_argument("--k", default=10, type=int, 
                        help="Top-k value for calculating Top-k accuracy")

    parser.add_argument("--margin", default=100.0, type=float, 
                        help="Margin for the rank loss")
    parser.add_argument("--output_dir", default="",
                        help="path where to save, empty for no saving")
    return parser


def main(args: argparse.Namespace) -> None:

    dataset = TrainDataset()
    train_size = int(args.split_ratio * len(dataset))
    eval_size = len(dataset) - train_size
    dataset_train, dataset_eval = random_split(
        dataset=dataset, 
        lengths=[train_size, eval_size]
        )

    data_loader_train = DataLoader(
        dataset=dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=train_collate_fn)
    data_loader_eval = DataLoader(
        dataset=dataset_eval, 
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_collate_fn)

    model = SynthesisPredictionModel()

    optimizer = Adam(params=model.parameters(), lr=args.lr)
    criterion = CustomRankLoss(margin=args.margin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = Path(args.output_dir)
    writer = SummaryWriter()

    print("Start training")
    start_time = time.time()

    for epoch in range(args.epochs):

        avg_loss = train_one_epoch(
            model, data_loader_train, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Complete. "
              f"Average Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        elapsed_time = time.time() - start_time
        elap_str = str(datetime.timedelta(seconds=elapsed_time))
        print(f"Time elapsed: {elap_str}")

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint every 100 epochs
            if (epoch + 1) % 100 == 0:
                checkpoint_paths.append(
                    output_dir / f"checkpoint{epoch:04}.pth"
                    )
            for checkpoint_path in checkpoint_paths:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }, checkpoint_path)

        eval_mrr, eval_top_k = evaluate_model(model, data_loader_eval, device,
                                               k=args.k)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Evaluation Complete.")
        print(f"Average MRR: {eval_mrr:.4f}.")
        print(f"Average Top-10 Accuracy: {eval_top_k:.4f}")
        writer.add_scalar("MRR/eval", eval_mrr, epoch)
        writer.add_scalar("Top-10/eval", eval_top_k, epoch)
    writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Total Training time {}".format(total_time_str))
    writer.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training and evaluation script", 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
