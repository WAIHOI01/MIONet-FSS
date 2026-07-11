"""M1 MIONet training entry point."""
import argparse, os
import torch
from torch.utils.data import DataLoader
from common.config import DEVICE, DATASET_PATH
from common.mionet import MIONet, PhysicsBranch
from common.dataset import ForwardDualModeThetaMagDataset, load_pair_splits, build_freq_tensor
from common.loss import DualModePhysicsLoss


def train(args):
    data_obj, train_pairs, val_pairs = load_pair_splits(
        args.dataset or DATASET_PATH, train_ratio=0.8, seed=42,
        theta_min_deg=0, theta_max_deg=60, split_by_geometry=True)
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_ds = ForwardDualModeThetaMagDataset(data_obj, train_pairs)
    val_ds = ForwardDualModeThetaMagDataset(data_obj, val_pairs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = MIONet(order=16, feat_dim=512).to(DEVICE)
    criterion = DualModePhysicsLoss(phys_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    best_val = float("inf")
    os.makedirs(args.output, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for geom, (te_s11, te_s21, tm_s11, tm_s21) in train_loader:
            geom = geom.to(DEVICE)
            te_s11, te_s21 = te_s11.to(DEVICE), te_s21.to(DEVICE)
            tm_s11, tm_s21 = tm_s11.to(DEVICE), tm_s21.to(DEVICE)
            B = geom.shape[0]
            freq = build_freq_tensor(data_obj.norm_freqs, DEVICE).repeat(B, 1)
            enc_te = PhysicsBranch.encode_physics(
                torch.zeros(B, device=DEVICE), torch.zeros(B, device=DEVICE),
                torch.zeros(B, device=DEVICE, dtype=torch.long))
            enc_tm = PhysicsBranch.encode_physics(
                torch.zeros(B, device=DEVICE), torch.zeros(B, device=DEVICE),
                torch.ones(B, device=DEVICE, dtype=torch.long))
            p_te_s11, p_te_s21 = model(geom, freq, enc_te)
            p_tm_s11, p_tm_s21 = model(geom, freq, enc_tm)
            loss = criterion(p_te_s11, p_te_s21, p_tm_s11, p_tm_s21,
                             te_s11, te_s21, tm_s11, tm_s21)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for geom, (te_s11, te_s21, tm_s11, tm_s21) in val_loader:
                geom = geom.to(DEVICE)
                te_s11, te_s21 = te_s11.to(DEVICE), te_s21.to(DEVICE)
                tm_s11, tm_s21 = tm_s11.to(DEVICE), tm_s21.to(DEVICE)
                B = geom.shape[0]
                freq = build_freq_tensor(data_obj.norm_freqs, DEVICE).repeat(B, 1)
                enc_te = PhysicsBranch.encode_physics(
                    torch.zeros(B, device=DEVICE), torch.zeros(B, device=DEVICE),
                    torch.zeros(B, device=DEVICE, dtype=torch.long))
                enc_tm = PhysicsBranch.encode_physics(
                    torch.zeros(B, device=DEVICE), torch.zeros(B, device=DEVICE),
                    torch.ones(B, device=DEVICE, dtype=torch.long))
                p_te_s11, p_te_s21 = model(geom, freq, enc_te)
                p_tm_s11, p_tm_s21 = model(geom, freq, enc_tm)
                loss = criterion(p_te_s11, p_te_s21, p_tm_s11, p_tm_s21,
                                 te_s11, te_s21, tm_s11, tm_s21)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.output, "best_model.pth"))
        if epoch % 10 == 0:
            print(f"epoch={epoch:03d}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f}")
    print(f"Done. Best val: {best_val:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dataset", default=None)
    p.add_argument("--output", default="./checkpoints")
    train(p.parse_args())
