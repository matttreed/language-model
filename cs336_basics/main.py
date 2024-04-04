from configs.config import get_config

def main():
    config = get_config()["model"]["embedding_size"]
    print(config)

if __name__ == '__main__':
    main()