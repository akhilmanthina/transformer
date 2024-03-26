from collections import defaultdict
import string


def encode(text, merge_rules, vocab):
    pretokenized_text = pretokenize([text])
    encoded_text = [[c for c in word] for word in pretokenized_text]
    #print(encoded_text)
    token_to_id = {token: i for i, token in enumerate(vocab)}

    #print(merge_rules)
    #take each merge rule sequentially and apply it to the encoded text
    for pair, merge in merge_rules.items():
        for idx, word in enumerate(encoded_text):
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    #print("merging")
                    new_word.append(merge)
                    i += 2
                else:
                    #print("appending")
                    new_word.append(word[i])
                    #print(new_word)
                    i += 1

            encoded_text[idx] = new_word
    #print(encoded_text)
    
    return [token_to_id[item] for sublist in encoded_text for item in sublist]
    


def decode(encoded_tokens, merge_rules, vocab):
    #NOTE: Requires python 3.7 or higher to maintain insertion order of merge_rules
    id_to_token = {i: token for i, token in enumerate(vocab)}
    decoded_tokens = [id_to_token[token] for token in encoded_tokens]
    reversed_merge_rules = {v: k for k, v in reversed(merge_rules.items())}
    i = 0
    while i < len(decoded_tokens):
        for merged, original in reversed_merge_rules.items():
            if i < len(decoded_tokens) - 1 and decoded_tokens[i] == original[0] and decoded_tokens[i+1] == original[1]:
                decoded_tokens[i:i+2] = [merged]
                # Adjust index since we've replaced two tokens with their merged form
                i -= 1  # Potentially adjust based on merging behavior
        i += 1
    
    # Reconstruct the original text
    reconstructed_text = ''.join(decoded_tokens).replace('Ġ', ' ')
    return reconstructed_text


def pretokenize(input):
    #word_frequency = defaultdict(int)
    pretokenized = []

    #Pretokenize the corpus based on gpt2 pretokenization
    for text in input:
        is_num = False
        prev_space = False
        current_word = ""

        for char in text:
            #if the character is alphanumeric, add it to the current word
            if char.isalpha():
                #start a new word if the prev word was a number
                if is_num:
                    if current_word:
                        pretokenized.append(current_word)
                    current_word = ""
                    is_num = False
                #replicate gpt2 behavior, if the word is lowercase, add a leading character to show middle of sentence
                if (current_word == "" and char.islower()) or (current_word == "" and prev_space):
                    current_word += "Ġ"

                current_word += char
                prev_space = False
            
            if char.isdigit():
                #start a new word if prev word was a letter
                if not is_num:
                    if current_word:
                        pretokenized.append(current_word)
                    current_word = ""

                current_word += char
                is_num = True
                prev_space = False

            #replicate gpt2 behavior, punctuation starts a new word
            elif char in string.punctuation:
                if current_word:
                    pretokenized.append(current_word)
                current_word = char
                is_num = False
                prev_space = False
            
            #whitespace starts a new word
            elif char.isspace():
                if current_word:
                    pretokenized.append(current_word)
                current_word = ""
                is_num = False
                prev_space = True
            
        pretokenized.append(current_word)

    #print(pretokenized)
    return pretokenized

def calculate_word_frequency(pre_tokenized):
    word_frequency = defaultdict(int)
    for token in pre_tokenized:
        word_frequency[token] += 1
    return word_frequency


def generate_vocab(word_frequency):
    #starting vocab based on single characters
    alphabet = []
    words = word_frequency.keys()
    for word in words:
        for char in word:
            if char not in alphabet:
                alphabet.append(char)
    return alphabet

def count_pairs(word_frequency, word_splits):
    #count pairs based on how words are currently split
    pair_frequency = defaultdict(int)
    for word, freq in word_frequency.items():
        word = word_splits[word]
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_frequency[pair] += freq
    
    return pair_frequency

def merge_pair(pair, word_splits):
    #combine pair into a single token
    pair = pair[0] + pair[1]
    for key, split in word_splits.items():
        if len(split) < 2:
            continue
        
        new_tok = []
        i = 0
        while i < len(split) - 1:
            if split[i] + split[i+1] == pair:
                #replace the pair with the new token
                new_tok.append(pair)
                i += 2
            else:
                new_tok.append(split[i])
                i += 1
        
        #merge didn't happen at the end, add the last character
        if i == len(split) - 1:
            new_tok.append(split[-1])
        word_splits[key] = new_tok
    
    return word_splits
        


def train(corpus):
    word_frequency = calculate_word_frequency(pretokenize(corpus))
    #print(word_frequency)
    vocab = generate_vocab(word_frequency)
    merge_rules = {}

    max_vocab_size = 50
    #used to keep track of how each word is split as we combine pairs into tokens
    word_splits = {word: [c for c in word] for word in word_frequency.keys()}
    while len(vocab) < max_vocab_size:
        pair_frequency = count_pairs(word_frequency, word_splits)
        most_frequent_pair = max(pair_frequency.items(), key=lambda x: x[1])
        vocab.append(most_frequent_pair[0][0] + most_frequent_pair[0][1])
        word_splits = merge_pair(most_frequent_pair[0], word_splits)
        merge_rules[most_frequent_pair[0]] = vocab[-1]
    
    return vocab, merge_rules


if __name__ == "__main__":
    corpus = ["Hello world, how are you doing today? I am doing well!",
              "I enjoy coding in python, it is a fun language to work with.",
              "I am currently working on a project that involves NLP.",
              "This is a NLP tokenizer that I am working on. I hope it works well!",
              "Byte pair encoding is a method of tokenization that is used in NLP.",
              "NLP NLP NLP"
    ]
    

    vocab, merge_rules = train(corpus)
    #print(encode("I love doing work in NLP!", merge_rules, vocab))
    encoded = encode("I love doing work in NLP!", merge_rules, vocab)
    decoded = decode(encoded, merge_rules, vocab)
    print(encoded)
    print(decoded)