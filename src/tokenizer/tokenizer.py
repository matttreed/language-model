import regex as re
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Iterator
from dataclasses import dataclass
from abc import ABC
import json
import base64
from tqdm import tqdm
import time
from memory_profiler import memory_usage
import psutil
import os

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}{1,8}| ?\p{N}{1,8}| ?[^\s\p{L}\p{N}]{1,8}|\s{1,8}(?!\S)|\s+"""

NUM_START_TOKENS = 256
CHUNK_SIZE = 1000 * 1024 * 1024  # 1 GB

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


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams, special_tokens=None):
        self.params = params
        self.special_tokens = special_tokens
        self.reverse_vocab = {v: k for k, v in self.params.vocab.items()} # find some way to do "is prefix"
        self.max_len_token = max([len(word) for word in self.params.vocab.values()]) # 128 

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        def encode_item_to_bytes(item):
            # For byte escape sequences (e.g., "\\xf8"), convert them to actual bytes
            RE = r"(\\x[0-9a-fA-F]{2})"
            item_list = [string for string in re.split(RE, item) if len(string)]
            b = b""
            for item in item_list:
                if item.startswith("\\x"):
                    b += bytes([int(item[2:], 16)])
                else:
                    b += item.encode("utf-8")
            return b

        with open(vocab_filepath, 'r') as file:
            vocab = {int(num): encode_item_to_bytes(b) for b, num in json.load(file).items()}
            # TODO -> load as b"\xe2" instead of b"\\xe2"


        with open(merges_filepath, 'r') as file:
            merges: Dict[Tuple[int, int], int] = {}
            for line in file:
                merge = line.rstrip().split(" ")
                merges[(int(merge[0]), int(merge[1]))] = int(merge[2])
        params = BPETokenizerParams(vocab, merges)
        return cls(params, special_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def encode(self, text: str) -> List[int]:
        split_text = [text]
        if self.special_tokens: # split by special tokens first
            special_split = r"(" + r"|".join(re.escape(tok) for tok in sorted(self.special_tokens, reverse=True)) + r")" #+  PAT
            split_text: List[str] = [string for string in re.split(special_split, text) if len(string)] # get rid of empty strings

        pretokenized_text: List[List[bytes]] = [] # list of list of bytes. inner lists are mostly just individual bytes except special tokens which are already fully formed
        # print(self.reverse_vocab)
        for t in split_text:
            if self.special_tokens and t in self.special_tokens:
                pretokenized_text.append([self.reverse_vocab[t.encode("utf-8")]])
            else:
                list_of_bytes: List[bytes] = [string.encode("utf-8") for string in re.findall(PAT, t)]
                list_of_list_of_bytes: List[List[bytes]]= [[self.reverse_vocab[bytes([b])] for b in bs] for bs in list_of_bytes]
                pretokenized_text += list_of_list_of_bytes

        inds: List[int] = [] # token numbers

        for i, token in tqdm(enumerate(pretokenized_text)):
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


    print("Reading file")
    with open(input_path, 'r') as file:
        text = file.read()

    text_len = len(text)
    print("Length of text:", text_len)
    # pretokenized_text = re.findall(PAT, text)
    pretokenized_text = []
    with tqdm(total=text_len, desc="PreTokenizing") as pbar:
        for pretoken in re.finditer(PAT, text):
            string = pretoken.group()
            pretokenized_text.append(string)
            pbar.update(len(string))

    token_counts = defaultdict(int)
    for token in tqdm(pretokenized_text, desc="Counting tokens"):
        token_ints = tuple(token.encode("utf-8"))
        token_counts[token_ints] += 1

    pairs = defaultdict(int)
    for token, count in tqdm(token_counts.items(), desc="Counting pairs"):
        for i in range(len(token) - 1):
            pair = token[i:i+2]
            pairs[pair] += count

    pbar = tqdm(total=vocab_size - len(vocab), desc="Building Vocab")
    while len(vocab) < vocab_size:
        pbar.update(1)

        if not pairs:
            break

        best_pair = max(pairs, key=lambda pair: (pairs[pair], str(vocab[pair[0]]) + str(vocab[pair[1]]))) # (int, int)

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


def save_bpe_params(params: BPETokenizerParams, vocab_filepath: str, merges_filepath: str):

    with open(vocab_filepath, 'w') as file:
        dictionary = {b.decode('utf-8', errors="backslashreplace"): num for num, b in params.vocab.items()}

        json.dump(dictionary, file)
        # json.dump({f"\u{b[0]:04x}": num for num, b in params.vocab.items()}, file)
        # encoded_vocab = {f"\\u{b[0]:04x}": num for num, b in params.vocab.items()}
        # with open(vocab_filepath, 'w', encoding='utf-8') as file:
        # # Directly writing the string representation to avoid JSON's escaping
        #     file.write(json.dumps(encoded_vocab))
    with open(merges_filepath, 'w') as file:
        for pair, new_index in params.merges.items():
            file.write(f"{pair[0]} {pair[1]} {new_index}\n")

def train_tokenizer_from_data():
    # data_path = "data/raw/TinyStoriesV2-GPT4-train.txt"
    # vocab_filepath="src/tokenizer/saved/tiny_stories_vocab.json"
    # merges_filepath="src/tokenizer/saved/tiny_stories_merges.txt"
    data_path = "data/raw/owt_train_2G.txt"
    # data_path = "/data/owt_train_2G.txt"
    vocab_filepath="src/tokenizer/saved/owt_vocab.json"
    merges_filepath="src/tokenizer/saved/owt_merges.txt"
    # data_path = "data/test.txt"
    # vocab_filepath="cs336_basics/outputs/test_vocab.json"
    # merges_filepath="cs336_basics/outputs/test_merges.txt"

    result = train_bpe(data_path, 32000, ["<|endoftext|>"])
    save_bpe_params(result, vocab_filepath, merges_filepath)

if __name__ == "__main__":
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)

    train_tokenizer_from_data()

    end_time = time.time()
    final_memory = process.memory_info().rss / (1024 * 1024)

    time_taken = end_time - start_time
    memory = final_memory - initial_memory

    print("time_taken: ", time_taken)
    print("max_memory (MB): ", memory)
    # train_tokenizer_from_data()
    # tokenizer = BPETokenizer.from_files("cs336_basics/outputs/tiny_stories_vocab.json", "cs336_basics/outputs/tiny_stories_merges.txt", ["<|endoftext|>"])
    # test = "ⵡ◌⫭∿⾯⬃⁙⦨⑄ⱗ⡇⽷⿷⍱␹⎸⛘⹷⩩<|endoftext|>Ⓟ⻉ⷤ⊘⡪▋ℤⷛ⑈≏◣ⴌ⡨⣭⾷⃘⍓ℶ ⹣◤⪧┊℻⊤⬣⛯⋦⢯<|endoftext|> <|endoftext|>⏀⽒⪤⸖ⴘ⍕⍽ⁿ⃮⼢≢⹭⠂≙⤯Ⅶ<|endoftext|><|endoftext|>⟀⌧⏓≋ⶴ⨖ⓧ⫚⪄⥠⌚⪙⟟␒⯅✴⸁⼱℡⴨⥶┺⣟✦ⲍ␞⫒⢣Ⳓ␕⦵⍟⸇▘⓹⧇"

    # print(gpt2_bytes_to_unicode())

    # print(list(tokenizer.params.vocab.items())[:300])
    # encoded = tokenizer.encode(test)
    # decoded = tokenizer.decode(encoded)
    # print(test)
    # print(decoded)
    # print(encoded)
    # print(test == decoded)

    # time_taken:  522.8188726902008
    # max_memory (MB):  47.75