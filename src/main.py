from src.training.train import train_model
from src.testing.test import sample_from_model

def main():
    train_model("0.0", from_checkpoint_k=0)

    # prompt = "hello this is me doing a test and I will continue here "
    # sample_from_model(prompt, "0.0", from_checkpoint_k=9, max_tokens=400)

    # with open("data/raw/owt_train.txt", "r") as f:
    #     text = f.read()
    
    # with(open("data/raw/owt_2G.txt", "w")) as f:
    #     f.write(text[:2000000000])

if __name__ == '__main__':
    main()