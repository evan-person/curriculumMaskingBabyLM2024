import torch

from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# load vocab size from tokenizer
tokenizer_name = "./tokenizer_eli5"
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
    file_path="../eli5data/10m_train.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import random
from helper_scripts import pos_tag_sentence, get_pos_tag, print_token_ids, mask_word_and_tokens_with_weights_and_percentage, create_pos_tag_weights, create_weights_from_percentage

class CustomMaskCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, pos_tags, **kwargs):
        # pos_tags is a list of the parts of speech that can be tokenized
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.pos_tags = pos_tags
    
    def __call__(self, examples):
        # Pad the batch of sequences and create attention masks and word ids
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        # iterate through the batch and mask the input
        for i, sentence in enumerate(batch["input_ids"]):
            sentence_words = self.tokenizer.decode(sentence)
            pos_tagged_sentence = nltk.pos_tag(word_tokenize(sentence_words))
            tokenized_sentence = sentence
            word_id_list = self.tokenizer.batch_encode_plus([sentence_words], return_tensors="pt").word_ids(batch_index=0)
            labels = torch.full_like(sentence, -100)
        

            # Find the indices of words matching the POS tags
            target_words = [i for i, (word, tag) in enumerate(pos_tagged_sentence) if tag in self.pos_tags]

           # Determine the number of words to mask
            num_words_to_mask = min(len(target_words),max(1, int(len(sentence_words) * self.mlm_probability))) #previously len(target_words)

            # Randomly select words to mask
            words_to_mask = set(random.sample(target_words, num_words_to_mask))
            
            # print(words_to_mask)
            # print(len(tokenized_sentence))
            # print(len(word_id_list))
            # word_mask = torch.zeros(len(tokenized_sentence))

            for j, word in enumerate(words_to_mask):
                # word_mask[word_id_list[word]] = 1
                if word < len(word_id_list):
                    
                    labels[word_id_list[word]] = sentence[word_id_list[word]]
                    sentence[word_id_list[word]] = self.tokenizer.mask_token_id
            
            batch["labels"][i] = labels
            batch["input_ids"][i] = torch.tensor(sentence)
            # batch["labels"][i] = torch.tensor(labels)

    
        # Return masked input ids and labels
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"]
        }


# include the following: UH, NN, PRP, NNS, NNP, VB, VBZ, VBP, IN, JJ, PRP$, NNPS,WDT,WP, DT,  VBP, VBG, VBN, VBZ, POS, RB, RBR, RBS, RP, JJR, JJS, MD
pos_tags = ['UH','NN','PRP','CC','NNS', 'NNP', 'VB', 'VBZ', 'VBP', 'IN', 'JJ', 'PRP$', 'NNPS','WDT','WP', 'DT',  'VBP', 'VBG', 'VBN', 'VBZ', 'POS', 'RB', 'RBR', 'RBS', 'RP', 'JJR', 'JJS', 'MD']
pos_schedule = [2,3,8,14,19,27,31]
epoch_schedule = [1,2,3,3,3,5,10]

from transformers import Trainer, TrainingArguments

import wandb

wandb.init(

    project="babyLM",
    name="roberta_10m_eli5_curriculum_masking",
    entity="prime-lab-mtu",
    config={
        "batch_size": 512,
        "epochs":30,
        "mlm_probability": 0.15,
        "model": "roberta",
        "dataset": "eli5_10m",
        "learning_rate": 5e-5,
    },
)

for i,step in enumerate(pos_schedule):
        
    training_args = TrainingArguments(
        output_dir="./roberta_10m_eli5_curriculum_masking",
        overwrite_output_dir=True,
        num_train_epochs=epoch_schedule[i],
        per_device_train_batch_size=256,
        report_to="wandb",
        logging_steps=100,
        save_steps=1_000,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        learning_rate=1e-4,
        # warmup_ratio=0.1,
        prediction_loss_only=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=CustomMaskCollator(
            pos_tags=pos_tags[:step],
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.2,
        ),
        train_dataset=dataset,
        
    )







    trainer.train()


trainer.save_model("./roberta_10m_eli5_curriculum_masking")


