import argparse
import time
import os
import math

import torch
import wandb
from torch_geometric.loader import DataLoader
from pypower.api import case30
from algorithm import (
    gcncase30,
    gatcase30,
    gat_cnn_case30,
    cnn_transformer_locator,
    baseline_gae_30,
)
from utils import find_edge_indices_within_nodes, split_dataset_randomly, GNN_dataset, plot_curve, \
    BFS_algorithm
from tqdm import tqdm
from torch_geometric.transforms import BaseTransform


class AddGaussianNoise(BaseTransform):
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, data):
        data.x = data.x + torch.randn_like(data.x) * self.std
        return data


def build_model(model_name, N, E_H, V_H_zero, edge_index, device):
    """返回模型、保存路径及用于wandb记录的结构超参。"""
    if model_name == 'gat':
        config = {
            "arch_type": "gat",
            "in_features": 3,
            "gat_channels": [256, 256, 256],
            "gat_heads": [4, 4, 4],
            "mlp_dims": [256, 128, 128],
            "dropout": 0.2,
        }
        return (
            gatcase30(
                in_features=config["in_features"],
                gat_channels=tuple(config["gat_channels"]),
                gat_heads=tuple(config["gat_heads"]),
                mlp_dims=tuple(config["mlp_dims"]),
                dropout=config["dropout"],
                num_nodes=N,
                e_h=E_H,
                edge_index=edge_index
            ).to(device),
            f"model/case30_gat_baseline.pth",
            config
        )
    elif model_name == 'gat_cnn':
        config = {
            "arch_type": "gat_cnn",
            "in_features": 3,
            "gat_channels": [256, 256, 256],
            "gat_heads": [4, 4, 4],
            "dropout": 0.2,
        }
        return (
            gat_cnn_case30(
                in_features=config["in_features"],
                gat_channels=tuple(config["gat_channels"]),
                gat_heads=tuple(config["gat_heads"]),
                dropout=config["dropout"],
                num_nodes=N,
                v_h=V_H_zero,
                e_h=E_H,
                edge_index=edge_index
            ).to(device),
            f"model/case30_gat_cnn.pth",
            config
        )
    elif model_name == 'gcn':
        config = {
            "arch_type": "gcn",
            "in_features": 3,
            "gcn_channels": [256, 256, 256],
            "mlp_dims": [256, 128, 128],
            "dropout": 0.2,
        }
        return (
            gcncase30(
                in_features=config["in_features"],
                gcn_channels=tuple(config["gcn_channels"]),
                mlp_dims=tuple(config["mlp_dims"]),
                dropout=config["dropout"],
                num_nodes=N,
                v_h=V_H_zero,
                e_h=E_H,
                edge_index=edge_index
            ).to(device),
            f"model/case30_gcn_baseline.pth",
            config
        )
    elif model_name == 'cnn_transformer':
        config = {
            "arch_type": "cnn_transformer",
            "in_features": 3,
            "mlp_out_features": len(E_H),
        }
        return (
            cnn_transformer_locator(
                in_features=config["in_features"],
                mlp_out_features=config["mlp_out_features"],
                num_nodes=N,
                e_h=E_H,
                edge_index=edge_index
            ).to(device),
            f"model/case30_cnn_transformer.pth",
            config
        )
    elif model_name == 'gae':
        config = {
            "arch_type": "gae",
            "in_features": 3,
            "mlp_out_features": len(E_H),
            "dropout": 0.2,
        }
        return (
            baseline_gae_30(
                in_features=config["in_features"],
                mlp_out_features=config["mlp_out_features"],
                num_nodes=N,
                e_h=E_H,
                edge_index=edge_index,
                dropout=config["dropout"]
            ).to(device),
            f"model/case30_gae.pth",
            config
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gat", "gat_cnn", "gcn", "cnn_transformer", "gae"], default="gat")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr_step_size", type=int, default=40)  # 对齐 train_gat.py：20 轮后开始衰减
    parser.add_argument("--lr_gamma", type=float, default=0.5)   # 对齐 train_gat.py 衰减系数
    parser.add_argument("--loss_type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--drop_non_vh_ratio", type=float, default=0.1,
                        help="训练时对非 V_H 节点特征随机置零的比例")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mpc = case30()

    start_node = 6
    length_V_H = 8
    V_H = BFS_algorithm(mpc, start_node, length_V_H)

    N = mpc['bus'].shape[0]
    edge_index = torch.tensor((mpc['branch'][:, :2].astype(int) - 1).T, dtype=torch.long)
    E_H = find_edge_indices_within_nodes(mpc['branch'], V_H)
    graph_data = GNN_dataset(root='', edge_index=edge_index)
    V_H_zero = [int(v) - 1 for v in V_H]
    non_vh_indices = [i for i in range(N) if i not in V_H_zero]

    for data in graph_data:
        data.x[V_H_zero, 1:] = 0
                                                         
    torch.manual_seed(10)
    train_data, val_data, test_data = split_dataset_randomly(graph_data)

    wandb.init(
        project="gnn_case30_main",
        config={
            "model": args.model,
            "lr": args.lr,
            "batch": args.batch,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "V_H_length": length_V_H,
            "start_node": start_node,
            "loss_type": args.loss_type,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "num_nodes": N,
            "num_edges": edge_index.shape[1],
            "drop_non_vh_ratio": args.drop_non_vh_ratio,
        },
    )

    model, save_path, arch_config = build_model(args.model, N, E_H, V_H_zero, edge_index, device)
    wandb.config.update({
        **arch_config,
        "edge_h_size": len(E_H),
        "v_h_zero": V_H_zero,
        "save_path": save_path,
    }, allow_val_change=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    wandb.watch(model, log="all", log_freq=50)
    print(f"Training {args.model.upper()} Model... ")
    time.sleep(0.001)

    with tqdm(total=args.epochs, desc='Training Progress') as pbar:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for data in train_loader:
                data = data.to(device)
                if args.drop_non_vh_ratio > 0 and non_vh_indices:
                    # 按批次随机丢弃非 V_H 节点的特征
                    B = data.x.size(0) // N
                    x_view = data.x.view(B, N, -1)
                    non_vh_idx = torch.as_tensor(non_vh_indices, device=device)
                    if non_vh_idx.numel() > 0:
                        drop_k = max(1, int(math.ceil(non_vh_idx.numel() * args.drop_non_vh_ratio)))
                        drop_k = min(drop_k, non_vh_idx.numel())
                        chosen = non_vh_idx[torch.randperm(non_vh_idx.numel(), device=device)[:drop_k]]
                        x_view[:, chosen, :] = 0
                        data.x = x_view.view_as(data.x)
                optimizer.zero_grad()
                out = model(data)              # logits
                if args.loss_type == "focal":
                    probs = torch.sigmoid(out)
                    targets = data.y
                    pt = torch.where(targets == 1, probs, 1 - probs)
                    alpha_t = torch.where(targets == 1, torch.full_like(targets, args.focal_alpha),
                                          torch.full_like(targets, 1 - args.focal_alpha))
                    focal_loss = -alpha_t * ((1 - pt) ** args.focal_gamma) * torch.log(pt.clamp(min=1e-6))
                    loss = focal_loss.mean()
                else:
                    loss = bce_loss_fn(out, data.y)    # BCEWithLogitsLoss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    if args.drop_non_vh_ratio > 0 and non_vh_indices:
                        B = data.x.size(0) // N
                        x_view = data.x.view(B, N, -1)
                        non_vh_idx = torch.as_tensor(non_vh_indices, device=device)
                        if non_vh_idx.numel() > 0:
                            drop_k = max(1, int(math.ceil(non_vh_idx.numel() * args.drop_non_vh_ratio)))
                            drop_k = min(drop_k, non_vh_idx.numel())
                            chosen = non_vh_idx[torch.randperm(non_vh_idx.numel(), device=device)[:drop_k]]
                            x_view[:, chosen, :] = 0
                            data.x = x_view.view_as(data.x)
                    out = model(data)           # logits
                    if args.loss_type == "focal":
                        probs = torch.sigmoid(out)
                        targets = data.y
                        pt = torch.where(targets == 1, probs, 1 - probs)
                        alpha_t = torch.where(targets == 1, torch.full_like(targets, args.focal_alpha),
                                              torch.full_like(targets, 1 - args.focal_alpha))
                        focal_loss = -alpha_t * ((1 - pt) ** args.focal_gamma) * torch.log(pt.clamp(min=1e-6))
                        val_loss += focal_loss.mean().item()
                    else:
                        val_loss += bce_loss_fn(out, data.y).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            pbar.set_postfix({'Train Loss': f'{train_loss:.6f}', 'Val Loss': f'{val_loss:.6f}'})
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break
            pbar.update(1)
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model state (val loss: {best_val_loss:.6f})")

    plot_curve(train_losses, val_losses)

    print("Testing Model...")
    model.eval()
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False)
    predictions, labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)           # logits
            probs = torch.sigmoid(out)
            prediction = probs > 0.5
            predictions.append(prediction)
            labels.append(data.y)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    TP = torch.sum(predictions * labels).item()
    TN = torch.sum(torch.logical_not(predictions) * torch.logical_not(labels)).item()
    FP = torch.sum(predictions * torch.logical_not(labels)).item()
    FN = torch.sum(torch.logical_not(predictions) * labels).item()

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    missed_detection_rate = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    wandb.log({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_false_alarm_rate": false_alarm_rate,
        "test_missed_detection_rate": missed_detection_rate,
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    main()

