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
    # data = serealize_data(tokenizer, 'data/raw/TinyStoriesV2-GPT4-train.txt', 'data/processed/TinyStoriesV2-GPT4-train.npy')
    data = serealize_data(tokenizer, 'data/raw/test-train.txt', 'data/processed/test-train.npy')
    # print(np.load('data/processed/TinyStoriesV2-GPT4-valid.npy').shape)