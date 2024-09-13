import torch

from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# load vocab size from tokenizer
tokenizer_name = "./tokenizer_eli5_100m_tok50k"
# tokenizer = ByteLevelBPETokenizer(
#     f"{tokenizer_name}-vocab.json",
#     f"{tokenizer_name}-merges.txt",
# ) 

# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )

tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name, max_len=512)

config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)
# model = RobertaForMaskedLM.from_pretrained("./roberta_100m_eli5_tok50k") #uncomment for pre-trained continuation training

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../data/eli5data/100m_train.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

from transformers import Trainer, TrainingArguments

import wandb

wandb.init(

    project="babyLM",
    name="roberta_100m_eli5_tok50k_redo",
    entity="prime-lab-mtu",
    config={
        "batch_size": 64,
        "epochs":30,
        "mlm_probability": 0.15,
        "model": "roberta",
        "dataset": "eli5_100m",
        "learning_rate": 5e-5,
    },
)

training_args = TrainingArguments(
    output_dir="./roberta_100m_eli5_tok50k",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    report_to="wandb",
    logging_steps=500,
    save_steps=10_000,
    save_total_limit=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


trainer.train()


trainer.save_model("./roberta_100m_eli5_tok50k")


