#!/usr/bin/env python3
"""Profile where time is spent: data loading vs GPU compute."""
import sys, time, torch
sys.path.insert(0, ".")

from src.config import PROCESSED_DIR
from src.preprocessing.dataset import create_dataloaders
from src.reconstruction.model import AttentionUNet3D
from src.reconstruction.train import OceanEmbedLoss

print("Loading dataset...")
t0 = time.time()
train_loader, _ = create_dataloaders(
    train_surface=PROCESSED_DIR / "train_surface.nc",
    train_target=PROCESSED_DIR / "train_target.nc",
    batch_size=2, num_workers=0, temporal_window=5,
)
print(f"Dataset ready in {time.time()-t0:.1f}s")

model = AttentionUNet3D().cuda()
criterion = OceanEmbedLoss(n_depth_levels=15).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Profile 5 batches
data_times = []
fwd_times = []
bwd_times = []

it = iter(train_loader)
for i in range(5):
    t1 = time.time()
    inputs, targets = next(it)
    t2 = time.time()
    data_times.append(t2 - t1)

    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    t3 = time.time()
    outputs = model(inputs)
    losses = criterion(outputs, targets)
    t4 = time.time()
    fwd_times.append(t4 - t3)

    losses["_total_tensor"].backward()
    optimizer.step()
    t5 = time.time()
    bwd_times.append(t5 - t4)

    print(f"  Batch {i}: data={data_times[-1]:.2f}s  fwd={fwd_times[-1]:.2f}s  bwd={bwd_times[-1]:.2f}s  total={t5-t1:.2f}s")

print(f"\nAverages: data={sum(data_times)/len(data_times):.2f}s  fwd={sum(fwd_times)/len(fwd_times):.2f}s  bwd={sum(bwd_times)/len(bwd_times):.2f}s")
print(f"Est epoch time (2190 batches): {(sum(data_times)+sum(fwd_times)+sum(bwd_times))/len(data_times)*2190/60:.1f} min")
