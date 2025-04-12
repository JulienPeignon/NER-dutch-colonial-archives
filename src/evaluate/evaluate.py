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

            # CRF returns list of lists of predicted label indices
            predictions = model(input_ids, attention_mask)

            for i in range(len(predictions)):
                pred = predictions[i]
                true = labels[i].cpu().tolist()

                # attention_mask = 0 for padding → remove those positions
                true_clean = [
                    label_map[true[j]] for j in range(len(pred)) if true[j] != -100
                ]
                pred_clean = [
                    label_map[pred[j]] for j in range(len(pred)) if true[j] != -100
                ]

                all_labels.append(true_clean)
                all_preds.append(pred_clean)

    print("🔍 seqeval classification report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    f1 = f1_score(all_labels, all_preds)
    print(f"\nF1-score (micro): {f1:.4f}")
