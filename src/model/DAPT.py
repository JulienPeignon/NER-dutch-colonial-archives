from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def run_dapt(
    model_name="GroNLP/bert-base-dutch-cased",
    data_path="./data/tokenized/mlm_corpus.txt",
    output_dir="./outputs/bertje-dapt",
    epochs=3,
    batch_size=32,
    max_length=128,
    mlm_probability=0.15,
):
    """
    Run Domain-Adaptive Pretraining (DAPT).
    """
    # Load dataset
    data_files = {"train": data_path}
    dataset = load_dataset("text", data_files=data_files)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # MLM data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train and save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
