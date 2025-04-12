from transformers import get_scheduler
import torch


def get_optimizer_and_scheduler(
    model,
    train_loader,
    epochs,
    lr=5e-5,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    scheduler_type="linear",
):
    """
    Returns an AdamW optimizer and a learning rate scheduler.
    """
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)

    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    return optimizer, scheduler
