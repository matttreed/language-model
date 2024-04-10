from src.training.train import train_model
from src.testing.test import sample_from_model

def main():
    # train_model("0.0", from_checkpoint_k=1)
    sample_from_model("0.0", from_checkpoint_k=1)

if __name__ == '__main__':
    main()