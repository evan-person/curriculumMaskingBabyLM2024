from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

# paths = [str(x) for x in Path("../data/train_10M/").glob("*.train")]

paths = "../data/eli5data/100m_train.txt"

print(paths)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

print(paths)

# Customize training
tokenizer.train(files=paths, vocab_size=50_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])


# Save the tokenizer
tokenizer.save_model(".", "train100M_eli5_tokenizer")
