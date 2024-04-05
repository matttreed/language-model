import regex as re
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Iterator
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
    def __init__(self, params: BPETokenizerParams, special_tokens=None):
        self.params = params
        self.special_tokens = special_tokens
        self.reverse_vocab = {v: k for k, v in self.params.vocab.items()} # find some way to do "is prefix"
        self.max_len_token = max([len(word) for word in self.params.vocab.values()]) # 128 

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r') as file:
            vocab = {int(k): v.encode("utf-8") for k, v in json.load(file).items()}
        with open(merges_filepath, 'r') as file:
            merges = [tuple(vocab[int(b)] for b in line.rstrip().split(" ")) for line in file]
        params = BPETokenizerParams(vocab, merges)
        cls(params, special_tokens)
    # Construct and
        # return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your
        # BPE training code output) and (optionally) a list of special tokens. This method should accept
        # the following additional parameters:
        # vocab_filepath: str
        # merges_filepath: str
        # special_tokens: list[str] | None = None

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
        # Given an iterable of
        # strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required
        # for memory-efficient tokenization of large files that we cannot directly load into memory.

    def encode(self, text: str) -> List[int]:
        split_text = [text]
        if self.special_tokens: # split by special tokens first
            special_split = r"(" + r"|".join(re.escape(tok) for tok in sorted(self.special_tokens, reverse=True)) + r")" #+  PAT
            split_text: List[str] = [string for string in re.split(special_split, text) if len(string)] # get rid of empty strings

        pretokenized_text: List[List[bytes]] = [] # list of list of bytes. inner lists are mostly just individual bytes except special tokens which are already fully formed

        for t in split_text:
            if self.special_tokens and t in self.special_tokens:
                pretokenized_text.append([self.reverse_vocab[t.encode("utf-8")]])
            else:
                list_of_bytes: List[bytes] = [string.encode("utf-8") for string in re.findall(PAT, t)]
                list_of_list_of_bytes: List[List[bytes]]= [[self.reverse_vocab[bytes([b])] for b in bs] for bs in list_of_bytes]
                pretokenized_text += list_of_list_of_bytes

        inds: List[int] = [] # token numbers

        for i, token in enumerate(pretokenized_text):
            merges_to_perform = {} # index to order
            while True: # merging
                for i in range(len(token) - 1): # find all merges
                    curr_merge =(token[i], token[i+1])
                    if curr_merge in self.params.merges:
                        merges_to_perform[i] = self.params.merges[curr_merge]
                if merges_to_perform: # do first merge that appears in merges
                    best_merge_index = min(merges_to_perform, key=lambda x: merges_to_perform[x])
                    token[best_merge_index] = self.params.merges[token[best_merge_index], token[best_merge_index+1]]
                    token.pop(best_merge_index+1)
                    merges_to_perform.clear()
                else:
                    break
            inds += token
   
        return inds

    def decode(self, indices: List[int]) -> str:
        bytes_list: List[bytes] = [self.params.vocab[i] for i in indices] # list of every index converted to bytes
        text = b''.join(bytes_list).decode("utf-8", errors="replace") # join bytes into one string then decode
        return text



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

    tokenizer = BPETokenizer(params=result, special_tokens=["<|endoftext|>"])
    test = tokenizer.encode("s")
    test_2 = tokenizer.decode(test)
    print(test, test_2)

    print(vocab)
    # print(merges[92:100])
