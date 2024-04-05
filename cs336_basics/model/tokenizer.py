import regex as re
from collections import defaultdict
from typing import List, Tuple, Dict
from dataclasses import dataclass
from abc import ABC

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_START_TOKENS = 256

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: Dict[int, bytes]             # index -> bytes
    merges: Dict[Tuple[int, int], int]  # (index1, index2) -> new_index

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, indices: List[int]) -> str:
        raise NotImplementedError
    


# class CharacterTokenizer(Tokenizer):
#     """Represent a text as a sequence of Unicode code points."""
#     def encode(self, text: str) -> List[int]:
#         return list(map(ord, text))

#     def decode(self, indices: List[int]) -> str:
#         return "".join(map(chr, indices))


# def merge(indices: Tuple[int], pair: Tuple[(int, int)], new_index: int) -> Tuple[int]:
#     """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
#     new_indices = []
#     i = 0
#     while i < len(indices):
#         if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
#             new_indices.append(new_index)
#             i += 2
#         else:
#             new_indices.append(indices[i])
#             i += 1
#     return new_indices



class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, text: str) -> List[int]:
        indices = list(map(int, text.encode("utf-8")))
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: List[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        text = b"".join(bytes_list).decode("utf-8")
        return text


# def train_bpe_ekrglekrm(input_path: str, vocab_size: int, special_tokens: list[str])-> BPETokenizerParams:
#     # note("Start with the list of bytes of `text`.")
#     with open(input_path, 'r') as file:
#         text = file.read()
#         # pretokenized_text = re.findall(PAT, text)
#     indices = list(map(int, text.encode("utf-8")))

#     # index1, index2 => merged index
#     merges: Dict[Tuple[int, int], int] = {}

#     # index -> bytes
#     vocab: Dict[int, bytes] = {
#         x: bytes([x]) for x in range(256)
#     }

#     for i in range(vocab_size):
#         # note("Count the number of occurrences of each pair of tokens")
#         counts = defaultdict(int)
#         for pair in zip(indices, indices[1:]):  # For each adjacent pair
#             counts[pair] += 1

#         if not counts:
#             break
#         # note("Find the most common pair.")
#         pair = max(counts, key=lambda pair: (counts[pair], pair[::-1]))

#         # note("Merge that pair.")
#         new_index = 256 + i
#         merges[pair] = new_index
#         vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

#         # note(f"Merge {vocab[pair[0]]} {vocab[pair[1]]} -> {vocab[new_index]}")
#         indices = merge(indices, pair, new_index)

#         # note(f"Text: {list(map(vocab.get, indices))}")

#     return BPETokenizerParams(vocab=vocab, merges=merges)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> (Dict[int, bytes], List[Tuple[bytes, bytes]]):
        
    merges: Dict[Tuple[int, int], int] = {}
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(NUM_START_TOKENS)}
    for i, token in enumerate(special_tokens, NUM_START_TOKENS):
        vocab[i] = token.encode("utf-8")



    with open(input_path, 'r') as file:
        text = file.read()
        pretokenized_text = re.findall(PAT, text)


    token_counts = defaultdict(int)
    for token in pretokenized_text:
        # for special_token in special_tokens:
            # if special_token in token and token not in special_tokens:
            #     print(token)
            # token_counts[NUM_START_TOKENS + i] += 1
            # continue
        token_ints = tuple(token.encode("utf-8"))
        token_counts[token_ints] += 1

    pairs = defaultdict(int)
    for token, count in token_counts.items():
        for i in range(len(token) - 1):
            pair = token[i:i+2]
            pairs[pair] += count

    while len(vocab) < vocab_size:

        if not pairs:
            break

        def sort_function(pair):
            return pairs[pair], str(vocab[pair[0]]) + str(vocab[pair[1]])
        best_pair = max(pairs, key=sort_function) # (int, int)
        # sorted_pairs_1 = sorted(pairs.items(), key=lambda x: (pairs[x], x), reverse=True)
        # sorted_pairs = sorted(pairs.items(), key=sort_function, reverse=True)

        del pairs[best_pair]
        new_token_value = len(vocab)
        merges[best_pair] = new_token_value
        vocab[new_token_value] = vocab[best_pair[0]] + vocab[best_pair[1]]

        for old_token in list(token_counts.keys()):
            i = 0
            while i < len(old_token) - 1:
                count = token_counts[old_token]

                if old_token[i:i+2] == best_pair:
                    del token_counts[old_token]
                    new_token = old_token[:i] + (new_token_value,) + old_token[i+2:]
                    token_counts[new_token] = count

                    if i > 0:
                        left = old_token[i-1:i+1]
                        pairs[left] -= count
                        if pairs[left] == 0:
                            del pairs[left]
                        new_left = new_token[i-1:i+1]
                        pairs[new_left] += count
                        
                    if i < len(old_token) - 2:
                        right = old_token[i+1:i+3]
                        pairs[right] -= count
                        if pairs[right] == 0:
                            del pairs[right]
                        new_right = new_token[i:i+2]
                        pairs[new_right] += count

                    old_token = new_token

                i += 1

    return BPETokenizerParams(vocab=vocab, merges=merges)

if __name__ == "__main__":
    # vocab, merges = train_bpe("data/test.txt", 300, ["<|endoftext|>"])
    result = train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    # result = train_bpe("data/test.txt", 400, ["<|endoftext|>"])
    vocab = result.vocab
    merges = [tuple(vocab[b] for b in pair) for pair in result.merges.keys()]

    # print(vocab)
    # print(merges[92:100])
