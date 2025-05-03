def load_iob_data(path):
    sentences = []
    labels = []
    current_tokens = []
    current_labels = []

    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue

            parts = line.split("\t")
            token = parts[0]
            label = parts[1]

            current_tokens.append(token)
            current_labels.append(label)

    # Catch any last sentence not followed by empty line
    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)

    return sentences, labels


def load_text_from_iob_mlm(path):
    texts = []
    current_tokens = []

    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if current_tokens:
                    text = " ".join(current_tokens)
                    texts.append(text)
                    current_tokens = []
                continue

            parts = line.split("\t")
            token = parts[0]
            current_tokens.append(token)

    # Catch last sentence
    if current_tokens:
        text = " ".join(current_tokens)
        texts.append(text)

    return texts
