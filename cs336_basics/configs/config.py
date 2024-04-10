import json

VERSION = "1.0"

class TrainingConfig():
    def __init__(self, json_data):
        self.lr = json_data['optimizer']['lr']
        self.betas = json_data['optimizer']['betas']
        self.eps = json_data['optimizer']['eps']
        self.weight_decay = json_data['optimizer']['weight_decay']
        self.batch_size = json_data["data"]["batch_size"]

class TransformerConfig():
    def __init__(self, json_data):
        self.vocab_size = json_data["model"]['vocab_size']
        self.context_length = json_data["model"]['context_length']
        self.num_layers = json_data["model"]['num_layers']
        self.d_model = json_data["model"]['d_model']
        self.num_heads = json_data["model"]['num_heads']
        self.d_ff = json_data["model"]['d_ff']
        self.attn_pdrop = json_data["model"]['attn_pdrop']
        self.residual_pdrop = json_data["model"]['residual_pdrop']

class Config():
    def __init__(self, json_data):
        with open(f'configs/transformer_config_{VERSION}.json', 'r') as file:
            json_data = json.load(file)

        self.training = TrainingConfig(json_data)
        self.transformer = TransformerConfig(json_data)
        self.random_seed = json_data["random_seed"]
