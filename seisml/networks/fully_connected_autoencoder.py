import torch.nn as nn
import gin

@gin.configurable()
class FCAutoEncoder(nn.Module):
    def __init__(self, layers=[100, 50, 25, 15, 10], dropout=0.4):
        super(FCAutoEncoder, self).__init__()

        self.input_size = layers[0]
        self.embedding_size = layers[-1]

        # Encoder
        self.encoder = nn.Sequential()
        for l in range(1, len(layers)):
            self.encoder.add_module(f'enc_fc{l}', nn.Linear(layers[l-1], layers[l]))
            self.encoder.add_module(f'enc_bn{l}', nn.BatchNorm1d(layers[l]))
            self.encoder.add_module(f'enc_relu{l}', nn.ReLU())
            # self.encoder.add_module(f'enc_dropout{l}', nn.Dropout(dropout))

        # Decoder
        self.decoder = nn.Sequential()
        for l in reversed(range(1, len(layers))):
            self.decoder.add_module(f'dec_fc{l}', nn.Linear(layers[l], layers[l-1]))
            self.decoder.add_module(f'dec_bn{l}', nn.BatchNorm1d(layers[l-1]))
            self.decoder.add_module(f'dec_sig{l}', nn.Sigmoid())
            # self.decoder.add_module(f'dec_dropout{l}', nn.Dropout(dropout))


    def forward(self, x):
        embeddings = self.encoder(x)
        out = self.decoder(embeddings)

        return out, embeddings