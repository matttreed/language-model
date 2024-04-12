from src.tokenizer.tokenizer import BPETokenizer
import numpy as np

def serealize_data(tokenizer: BPETokenizer, data_path, save_path):
    print("Reading File")
    with open(data_path, 'r') as file:
        data = file.read()
    print("Tokenizing Data")
    tokenized_data = np.array(tokenizer.encode(data), dtype=np.int16)
    print("Saving Serialized Data")
    np.save(save_path, tokenized_data)
    # memmap = np.memmap(save_path, dtype=np.int64, mode='w+', shape=tokenized_data.shape)
    # memmap[:] = tokenized_data



if __name__ == "__main__":
    
    tokenizer = BPETokenizer.from_files("src/tokenizer/saved/tiny_stories_vocab.json", "src/tokenizer/saved/tiny_stories_merges.txt", ["<|endoftext|>"])
    # # data = serealize_data(tokenizer, 'data/raw/test.txt', 'data/processed/test.npy')

    print("TINY VALID")
    data = serealize_data(tokenizer, '/data/TinyStoriesV2-GPT4-valid.txt', 'data/processed/tiny_stories_valid.npy')
    print("TINY TRAIN")
    data = serealize_data(tokenizer, '/data/TinyStoriesV2-GPT4-train.txt', 'data/processed/tiny_stories_train.npy')



    tokenizer = BPETokenizer.from_files("src/tokenizer/saved/owt_vocab.json", "src/tokenizer/saved/owt_merges.txt", ["<|endoftext|>"])
    print("OWT VALID")
    data = serealize_data(tokenizer, '/data/owt_valid.txt', 'data/processed/owt_valid.npy')
    print("OWT TRAIN")
    data = serealize_data(tokenizer, '/data/owt_train.txt', 'data/processed/owt_train.npy')

    # data = serealize_data(tokenizer, 'data/raw/test-train.txt', 'data/processed/test-train.npy')
    # print(len(open('data/raw/TinyStoriesV2-GPT4-train.txt', 'r').read().split(" ")))