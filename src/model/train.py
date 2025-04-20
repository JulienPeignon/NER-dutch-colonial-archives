import os
import torch
from pathlib import Path
from typing import List, Tuple, Optional


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epochs: int = 50,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    clip_grad: bool = True,
    max_norm: float = 1.0,
    save_path: str = "best_model.pt",
) -> Tuple[List[float], List[float]]:
    """
    Train `model`, evaluate on `val_loader`, and save the parameters that achieve the
    lowest validation loss.

    Returns
    -------
    train_history : list[float]
        Average training loss for each epoch.
    val_history : list[float]
        Average validation loss for each epoch.
    """

    save_dir = Path(save_path).expanduser().resolve().parent
    if save_dir.name:  # no‑op if save_path is just a filename
        save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_history, val_history = [], []

    for epoch in range(epochs):
        # ------------------------------ training --------------------------- #
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, attention_mask, labels)
            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # ----------------------------- validation -------------------------- #
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for vbatch in val_loader:
                v_input_ids = vbatch["input_ids"].to(device)
                v_attention_mask = vbatch["attention_mask"].to(device)
                v_labels = vbatch["labels"].to(device)

                v_loss = model(v_input_ids, v_attention_mask, v_labels)
                running_val_loss += v_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1:02}/{epochs} | "
            f"Train loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}"
        )

        # --------------------------- model saving -------------------------- #
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            state_dict = (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            )

            checkpoint = {
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "model_state": state_dict,
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path)

        model.train()

    return train_history, val_history


def plot_train_val_loss(train_losses, val_losses, title="Training vs Validation Loss"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(
        ticks=range(0, len(train_losses), max(1, len(train_losses) // 20)),
        labels=[
            str(i + 1)
            for i in range(0, len(train_losses), max(1, len(train_losses) // 20))
        ],
        rotation=45,
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
