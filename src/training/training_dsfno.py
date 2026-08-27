import torch
from scipy.ndimage import convolve
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import amp
from typing import Callable, Optional, List, Tuple


def training_model(
    model: nn.Module,
    loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    learning_rate: float,
    epochs: int,
    train_loader: DataLoader
) -> Tuple[List[float], nn.Module]:
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    scaler = amp.GradScaler()

    loss_fn = loss if loss is not None else nn.MSELoss()
    for epoch in range(epochs):
        for high_res, low_res in train_loader:

            low_res = low_res.to(device, non_blocking=True)
            high_res = high_res.to(device, non_blocking=True)
            optimizer.zero_grad()

            with amp.autocast(device_type = 'cuda'):
                output = model(low_res)
                loss_value = loss_fn(output, high_res)
            
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss_value.item())

        print(f"Epoch {epoch + 1}/{epochs}, loss={loss_value.item():.6f}", end='\r')

    return losses, model
