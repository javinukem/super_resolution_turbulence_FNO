import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import amp
from typing import Callable, Optional, List, Tuple
import time


class EarlyStopper:  # from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def training_model(
    model: nn.Module,
    loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    learning_rate: float,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader = None,
    use_amp: bool = False,
    use_float16: bool = False,
    use_early_stopping: bool = False,
    early_stopping_patience: Optional[int] = None,
    early_stopping_delta: Optional[float] = None,
    print_updates: bool = False,
    accumulate_gradients: Optional[int] = None,
    gpu_id: Optional[int] = None,
    **model_kwargs,
) -> Tuple[List[float], nn.Module]:
    torch.cuda.empty_cache()
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_float16:
        model = model.half()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    if use_amp:
        scaler = amp.GradScaler()

    if test_loader is not None:
        test_losses = []

    loss_fn = loss if loss is not None else nn.MSELoss()
    early_stopper = None
    if use_early_stopping:
        patience = early_stopping_patience
        delta = early_stopping_delta
        if patience is None or delta is None:
            patience = 8
            delta = 0.0
        early_stopper = EarlyStopper(patience=patience, min_delta=delta)
    best_loss = float("inf")
    best_model = model

    train_loader_len = len(train_loader)
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        i = 0
        for data in train_loader:
            high_res = data[0]
            low_res = data[1]
            if use_float16:
                low_res = low_res.half()
                high_res = high_res.half()
            low_res = low_res.to(device, non_blocking=True)
            high_res = high_res.to(device, non_blocking=True)
            if use_amp:
                with amp.autocast(device_type="cuda"):
                    output = model(low_res, **model_kwargs)
                    loss_value = loss_fn(output, high_res)
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(low_res, **model_kwargs)
                loss_value = loss_fn(output, high_res)
                if accumulate_gradients is not None:
                    loss_value = loss_value / accumulate_gradients
                    loss_value.backward()
                    if ((i + 1) % accumulate_gradients == 0) or (
                        i + 1 == train_loader_len
                    ):
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    loss_value.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                i += 1
            epoch_loss += loss_value.detach()
        epoch_loss = epoch_loss.item() / len(train_loader)

        losses.append(epoch_loss)

        if test_loader is not None:
            model.eval()
            test_epoch_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    high_res = data[0]
                    low_res = data[1]
                    low_res = low_res.to(device, non_blocking=True)
                    high_res = high_res.to(device, non_blocking=True)
                    if use_float16:
                        low_res = low_res.half()
                        high_res = high_res.half()
                    output = model(low_res, **model_kwargs)
                    test_loss_value = loss_fn(output, high_res)
                    test_epoch_loss += test_loss_value.item()
            test_epoch_loss /= len(test_loader)
            test_losses.append(test_epoch_loss)
            if test_epoch_loss < best_loss:
                best_loss = test_epoch_loss
                best_model = model
            if early_stopper is not None:
                if early_stopper.early_stop(test_epoch_loss):
                    print("Early stopping")
                    break
        elapsed = time.time() - start_time
        if print_updates:
            if test_loader is not None:
                print(
                    f"Epoch {epoch + 1}/{epochs}, train_loss={epoch_loss:.6f}, test_loss={test_epoch_loss:.6f}, time={elapsed:.2f}s",
                    end="\r",
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs}, train_loss={epoch_loss:.6f}, time={elapsed:.2f}s",
                    end="\r",
                )
    return test_losses if test_loader else None, losses, best_model
