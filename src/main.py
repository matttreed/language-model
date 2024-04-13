from src.training.train import train_model
from src.testing.test import sample_from_model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train a model with float16 parameters.')
    parser.add_argument('--version', type=str, default=None, help='Version Number of Model')
    parser.add_argument('--train', action='store_true', help='Set training mode.')
    parser.add_argument('--checkpoint_k', type=int, default=None, help='Load model from checkpoint k.')
    parser.add_argument('--prompt', type=str, default="You are a helpful assistant. continue from here.", help='Prompt to generate text from.')
    parser.add_argument('--max_tokens', type=int, default=300, help='Load model from checkpoint k.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling.')
    parser.add_argument('--top_p', type=int, default=None, help='Use Top-p sampling for generation.')
    args = parser.parse_args()

    if args.train:
        train_model(args.version, from_checkpoint_k=args.checkpoint_k)

    else:
        text = sample_from_model(
            prompt=args.prompt,
            version=args.version, 
            from_checkpoint_k=args.checkpoint_k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
            )
        
        print(text)

if __name__ == '__main__':
    
    main()