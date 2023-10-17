# multi modal tokenizer
from zeta.tokenizers import TokenMonster


def tokenize(text):
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    tokens = tokenizer.tokenize(text)
    return tokens


def detokenize(tokens):
    tokenizer = TokenMonster("englishcode-32000-consistent-v1")
    text = tokenizer.detokenize(tokens)
    return text


tokenize = tokenize("Hello my name is Kye what's your name")
print(tokenize)
