from src.tokenizer.tokenizer import BPETokenizer
import numpy as np
import os

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

def serealize_data_iterator(tokenizer: BPETokenizer, data_path, save_path):
    print("Reading File")
    file_size = os.path.getsize(data_path)  # Get the size of the file
    halfway_point = file_size // 2
    with open(data_path, 'r') as file:
        file.seek(halfway_point)
        # data = file.read()
        print("Tokenizing Data")
        encoded = []
        iteration = 0
        for ind in tokenizer.encode_iterable(file):
            encoded.append(ind)
            if iteration % 100000000 == 0:
                print(f"Saving Iteration {iteration}, length {len(encoded)}")

                tokenized_data = np.array(encoded, dtype=np.int16)
                np.save(save_path, tokenized_data)
            iteration += 1
        tokenized_data = np.array(encoded, dtype=np.int16)
        print("Finished")
        np.save(save_path, tokenized_data)



if __name__ == "__main__":
    
    # tokenizer = BPETokenizer.from_files("src/tokenizer/saved/tiny_stories_vocab.json", "src/tokenizer/saved/tiny_stories_merges.txt", ["<|endoftext|>"])
    # # # data = serealize_data(tokenizer, 'data/raw/test.txt', 'data/processed/test.npy')

    # # print("TINY VALID")
    # # data = serealize_data(tokenizer, '/data/TinyStoriesV2-GPT4-valid.txt', 'data/processed/tiny_stories_valid.npy')
    # print("TINY TRAIN")
    # data = serealize_data(tokenizer, '/data/TinyStoriesV2-GPT4-train.txt', 'data/processed/tiny_stories_train.npy')



    # tokenizer = BPETokenizer.from_files("src/tokenizer/saved/owt_vocab.json", "src/tokenizer/saved/owt_merges.txt", ["<|endoftext|>"])
    # print("OWT VALID")
    # data = serealize_data(tokenizer, '/data/owt_valid.txt', 'data/processed/owt_valid.npy')
    # print("OWT TRAIN")
    # data = serealize_data_iterator(tokenizer, '/data/owt_train.txt', 'data/processed/owt_train_second_half.npy')

    # tokenizer = BPETokenizer.from_files("src/tokenizer/saved/tiny_stories_vocab.json", "src/tokenizer/saved/tiny_stories_merges.txt", ["<|endoftext|>"])
    # # # data = serealize_data(tokenizer, 'data/raw/test.txt', 'data/processed/test.npy')
    # with open("data/processed/owt_train.npy") as file:
    #     text = file.read()
    #     print(text)

    owt = np.load("data/processed/owt_train.npy")
    print(max(owt))
    owt_valid = np.load("data/processed/owt_valid.npy")
    print(max(owt_valid))
    tiny = np.load("data/processed/tiny_stories_train.npy")
    print(max(tiny))
    tiny_valid = np.load("data/processed/tiny_stories_valid.npy")
    print(max(tiny_valid))


    # print(test_inds.shape)
    # print(tokenizer.decode(test_inds))
    # print(len(tokenizer.decode(test_inds)), len(text))
    # print("TINY VALID")
    # data = serealize_data_iterator(tokenizer, 'data/raw/TinyStoriesV2-GPT4-valid.txt', 'data/processed/tiny_stories_valid.npy')
    # print("TINY TRAIN")
    # data = serealize_data(tokenizer, '/data/TinyStoriesV2-GPT4-train.txt', 'data/processed/tiny_stories_train.npy')


    # data = serealize_data(tokenizer, 'data/raw/test-train.txt', 'data/processed/test-train.npy')
    # print(len(open('data/raw/TinyStoriesV2-GPT4-train.txt', 'r').read().split(" ")))





    # first_half = np.load('data/processed/owt_train.npy')
    # second_half = np.load('data/processed/owt_train_second_half.npy')





    # tokenizer_owt = BPETokenizer.from_files("src/tokenizer/saved/owt_vocab.json", "src/tokenizer/saved/owt_merges.txt", ["<|endoftext|>"])
    # tokenizer_tiny = BPETokenizer.from_files("src/tokenizer/saved/tiny_stories_vocab.json", "src/tokenizer/saved/tiny_stories_merges.txt", ["<|endoftext|>"])

    # with open("data/raw/owt_10.txt") as file:
    #     owt = file.read()

    # with open("data/raw/tiny_10.txt") as file:
    #     tiny = file.read()

    # print("len owt bytes", len(owt.encode("utf-8")))
    # print("len owt encoded owt", len(tokenizer_owt.encode(owt)))
    # print("len owt encoded tiny", len(tokenizer_tiny.encode(owt)))

    # print("len tiny bytes", len(tiny.encode("utf-8")))
    # print("len tiny encoded owt", len(tokenizer_owt.encode(tiny)))
    # print("len tiny encoded tiny", len(tokenizer_tiny.encode(tiny)))
    # print(max(tokenizer_tiny.params.vocab.items(), key=lambda x: len(x[1])))

    # with open("data/raw/TinyStoriesV2-GPT4-train.txt") as file:
    #     owt = file.read()
    #     print(len(owt.encode("utf-8")))
    #     tokenizer_owt.encode(owt)