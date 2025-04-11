def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=5,
    log_steps=50,
):
    import torch

    global_step = 0
    train_history = []
    val_history = []

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            # Log training & validation every `log_steps`
            if (global_step % log_steps) == 0:
                # -- Compute validation loss --
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)
                        val_labels = val_batch["labels"].to(device)

                        val_logits = model(val_input_ids, val_attention_mask)
                        val_step_loss = criterion(
                            val_logits.view(-1, val_logits.shape[-1]),
                            val_labels.view(-1),
                        )
                        val_loss += val_step_loss.item()
                val_loss /= len(val_loader)

                # -- Store losses and print log --
                train_history.append(loss.item())
                val_history.append(val_loss)
                print(
                    f"Step {global_step} - "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )
                model.train()

    return train_history, val_history
