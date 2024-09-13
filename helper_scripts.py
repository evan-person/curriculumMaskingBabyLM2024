def mask_sentence(sentence):
    tokenized_sentence = tokenizer.tokenize(sentence)
    masked_sentence = []
    for token in tokenized_sentence:
        masked_sentence.append(token)
    return masked_sentence

# function to pos tag the sentence by using nltk
def pos_tag_sentence(sentence):
    # tokenized_sentence = tokenizer.tokenize(sentence)
    pos_tagged_sentence = nltk.pos_tag(word_tokenize(sentence))
    return pos_tagged_sentence

# function to get the pos tag of a word
def get_pos_tag(word):
    return nltk.pos_tag([word])[0][1]

# function to print the token ids of the sentence by using bert tokenizer
def print_token_ids(sentence):
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))



def mask_word_and_tokens_with_weights_and_percentage(sentence, pos_tag_weights, total_mask_percentage):
    # Tokenize the sentence
    words = word_tokenize(sentence)
    pos_tagged_sentence = nltk.pos_tag(words)
    tokenized_sentence = tokenizer.tokenize(sentence)
    
    # Create a list of words with their POS tag and their corresponding weights
    target_words = []
    for i, (word, tag) in enumerate(pos_tagged_sentence):
        if tag in pos_tag_weights:
            target_words.append((i, tag, pos_tag_weights[tag]))  # Append index, tag, and weight
    
    # Calculate total number of words to mask based on the overall percentage
    total_words_to_mask = max(1, int(len(target_words) * total_mask_percentage / 100))

    # Sort target words by their weights (higher weights first)
    target_words_sorted = sorted(target_words, key=lambda x: -x[2])

    # Randomly select words to mask, weighted by the given POS tags
    selected_words = random.sample(target_words_sorted, total_words_to_mask)
    words_to_mask = set(i for i, tag, weight in selected_words)
    
    masked_sentence = []
    masked_token_ids = []
    word_index = 0

    # Masking process
    for token in tokenized_sentence:
        if word_index < len(pos_tagged_sentence):
            word, tag = pos_tagged_sentence[word_index]

            if not token.startswith('##'):
                masking = (word_index in words_to_mask)

            if masking:
                masked_sentence.append('[MASK]')
                masked_token_ids.append(tokenizer.mask_token_id)
            else:
                masked_sentence.append(token)
                masked_token_ids.append(tokenizer.convert_tokens_to_ids(token))

            if not token.startswith('##'):
                word_index += 1
        else:
            masked_sentence.append(token)
            masked_token_ids.append(tokenizer.convert_tokens_to_ids(token))
    
    return masked_sentence, masked_token_ids

# create a function to create the pos tag weights

def create_pos_tag_weights(pos_tags, weights):
    pos_tag_weights = {}
    for tag, weight in zip(pos_tags, weights):
        pos_tag_weights[tag] = weight
    return pos_tag_weights

# function to create weights based on percentage for each pos tag
def create_weights_from_percentage(pos_tags1, percentage1, pos_tags2, percentage2):
    weights = []
    
    # Convert percentages to a fraction of 1
    total_percentage = percentage1 + percentage2
    percentage1_fraction = percentage1 / total_percentage
    percentage2_fraction = percentage2 / total_percentage

    # Create random weights for pos_tags1 and normalize them to sum to percentage1_fraction
    num_pos_tags1 = len(pos_tags1)
    pos_tags1_weights = [random.random() for _ in range(num_pos_tags1)]
    sum_pos_tags1_weights = sum(pos_tags1_weights)
    normalized_pos_tags1_weights = [(w / sum_pos_tags1_weights) * percentage1_fraction for w in pos_tags1_weights]
    
    # Append these normalized weights to the weights list
    weights.extend(normalized_pos_tags1_weights)
    
    # Create random weights for pos_tags2 and normalize them to sum to percentage2_fraction
    num_pos_tags2 = len(pos_tags2)
    pos_tags2_weights = [random.random() for _ in range(num_pos_tags2)]
    sum_pos_tags2_weights = sum(pos_tags2_weights)
    normalized_pos_tags2_weights = [(w / sum_pos_tags2_weights) * percentage2_fraction for w in pos_tags2_weights]
    
    # Append these normalized weights to the weights list
    weights.extend(normalized_pos_tags2_weights)
    weights = [round(w, 2) for w in weights]
    
    return weights
