import torch
import torch.nn as nn
import torch.nn.functional as F

def sir(cfg):
    return SIR(
        input_dim=cfg.MODEL.SIR.INPUT_DIM,
        hidden_dim=cfg.MODEL.SIR.HIDDEN_SIZE,
        num_heads=cfg.MODEL.SIR.NUM_HEADS,
        num_layers=cfg.MODEL.SIR.NUM_LAYERS,
        output_dim=cfg.MODEL.SIR.OUTPUT_DIM,
    )

def initialize_sir_parameters(m):
    if isinstance(m, nn.Linear):
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # Kaiming initialization for convolutional layers
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        # Initialize LayerNorm with ones for weight and zeros for bias
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        # Xavier initialization for attention layers
        nn.init.xavier_uniform_(m.in_proj_weight)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)

class SelfAttentionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(SelfAttentionTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim) for _ in range(num_layers)]
        )
        
    def forward(self, x):
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, feature_dim] -> [seq_len, batch_size, feature_dim]
        for layer in self.layers:
            x = layer(x)
        return x.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim] -> [batch_size, seq_len, feature_dim]

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(TransformerDecoder, self).__init__()
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim), num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, feature_dim] -> [seq_len, batch_size, feature_dim]
        x = self.transformer_decoder(x, x)
        x = self.fc(x)
        return x.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim] -> [batch_size, seq_len, feature_dim]

class SIR(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(SIR, self).__init__()
        self.self_attention_transformer = SelfAttentionTransformer(input_dim, hidden_dim, num_heads, num_layers)
        # self.transformer_decoder = TransformerDecoder(input_dim, hidden_dim, num_heads, num_layers, output_dim)
        # self.spatial_decoder = TransformerDecoder(input_dim, hidden_dim, num_heads, num_layers, output_dim)
        # self.feature_template = None  # init feature_template
        
    def forward(self, M):
        # M is a single input tensor with shape [batch_size, 2HW, C]
        # if self.feature_template is None:
        #     self.feature_template = torch.zeros_like(M).mean(dim=1, keepdim=True)  # init feature_template
        # print(f'feature template:{self.feature_template}')
        
        attention_output = self.self_attention_transformer(M)
        # global_feature = self.transformer_decoder(attention_output)
        
        # # update global_feature
        # global_feature = global_feature * 0.9 + self.feature_template * 0.1
        # print(f'global feature:{global_feature}')
        # # pass through spatial_decoder
        # integrated_features = self.spatial_decoder(global_feature)
        # print(f'intergrated features:{integrated_features}')
        # # update feature_template
        # self.feature_template = integrated_features * 0.1 + self.feature_template * 0.9
        
        return attention_output


def initialize_mlp_parameters(m):
    if isinstance(m, nn.Linear):
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MLPDecoder, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# MLP Decoder to possibly replace sir in the future
class MLP(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=1280, output_dim=320, num_layers=3):
        super(MLP, self).__init__()
        self.mlp = MLPDecoder(input_dim, hidden_dim, output_dim, num_layers)
        self.apply(initialize_mlp_parameters)

    def forward(self, M):
        # M is a tensor with shape [batch_size, 192, 5120]
        output = self.mlp(M)
        return output  # Output shape should be [batch_size, 192, 1280]

