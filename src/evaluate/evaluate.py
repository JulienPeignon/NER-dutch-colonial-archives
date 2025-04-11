import torch
from seqeval.metrics import classification_report, f1_score


def evaluate_ner_model(model, test_loader, label_map, device="cuda"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            for i in range(input_ids.size(0)):
                true = labels[i].cpu().tolist()
                pred = preds[i].cpu().tolist()

                true_clean = [label_map[l] for l, p in zip(true, pred) if l != -100]
                pred_clean = [label_map[p] for l, p in zip(true, pred) if l != -100]

                all_labels.append(true_clean)
                all_preds.append(pred_clean)

    print("🔍 seqeval classification report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    f1 = f1_score(all_labels, all_preds)
    print(f"\nF1-score (micro): {f1:.4f}")
