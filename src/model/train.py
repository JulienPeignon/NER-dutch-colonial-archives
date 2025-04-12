import torch
import os
import matplotlib.pyplot as plt


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=50,
    scheduler=None,
    clip_grad=True,
    max_norm=1.0,
    save_path="checkpoints/best_model.pt",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float("inf")
    train_history = []
    val_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, attention_mask, labels)
            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_input_ids = val_batch["input_ids"].to(device)
                val_attention_mask = val_batch["attention_mask"].to(device)
                val_labels = val_batch["labels"].to(device)

                val_loss = model(val_input_ids, val_attention_mask, val_labels)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

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
