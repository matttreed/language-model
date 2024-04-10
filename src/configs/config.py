import json

class TrainingConfig():
    def __init__(self, json_data):
        self.lr = json_data["training"]['lr']
        self.betas = json_data["training"]['betas']
        self.eps = json_data["training"]['eps']
        self.weight_decay = json_data["training"]['weight_decay']
        self.batch_size = json_data["training"]["batch_size"]
        self.num_iterations = json_data["training"]["num_iterations"]

class TransformerConfig():
    def __init__(self, json_data):
        self.context_length = json_data["model"]['context_length']
        self.num_layers = json_data["model"]['num_layers']
        self.d_model = json_data["model"]['d_model']
        self.num_heads = json_data["model"]['num_heads']
        self.d_ff = json_data["model"]['d_ff']
        self.attn_pdrop = json_data["model"]['attn_pdrop']
        self.residual_pdrop = json_data["model"]['residual_pdrop']

class DataConfig():
    def __init__(self, json_data):
        self.training_data = json_data["data"]['training_data']
        self.validation_data = json_data["data"]['validation_data']

class TokenizerConfig():
    def __init__(self, json_data):
        self.vocab_size = json_data["tokenizer"]['vocab_size']
        self.merges_filename = json_data["tokenizer"]['merges_filename']
        self.vocab_filename = json_data["tokenizer"]['vocab_filename']
        self.special_tokens = json_data["tokenizer"]['special_tokens']

class Config():
    def __init__(self, version):
        with open(f'src/configs/transformer_config_{version}.json', 'r') as file:
            json_data = json.load(file)

        self.training = TrainingConfig(json_data)
        self.transformer = TransformerConfig(json_data)
        self.data = DataConfig(json_data)
        self.tokenizer = TokenizerConfig(json_data)
        self.random_seed = json_data["random_seed"]
