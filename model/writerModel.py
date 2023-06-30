import torch
from torch import nn
import numpy as np
import math

from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).permute(1,0,2)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).permute(1,0,2)
    
class Model(nn.Module):
    def __init__(self, batch_size):
        super(Model, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.pos = PositionalEncoding(256, max_len=3000) # sequence length
        self.cls1 = nn.Parameter(torch.rand(1, 256)) # 1, enc_dim
        self.cls2 = nn.Parameter(torch.rand(1, 256))  
        self.bs = batch_size

    def forward(self, x, mask, indices):
#         x = torch.cat([self.cls1.repeat(4, 1).unsqueeze(1), x[:, :4, :], self.cls2.repeat(4, 1).unsqueeze(1),x[:,4:,:]], axis=1)
        # index = [3, 4, 2, 2]
        # pad_idx = [2,1,2,4]
        index = indices[0]
        pad_idx = indices[1]
        new_x = []
        
        # To insert class tokens
        for i in range(self.bs):
            new_x.append(torch.cat([self.cls1.unsqueeze(1), x[i, :index[i], :].unsqueeze(0), self.cls2.unsqueeze(1), x[i, index[i]: , :].unsqueeze(0)], axis=1))
            
            
        x = torch.cat(new_x)
        x = self.pos(x)
        
#         M = torch.ones((4,9,9)).to(torch.float32)  # Code to create n(=batch size = 4) masks
        M = []
        V = []
        with torch.no_grad():
            for i in range(self.bs):
                v1 = torch.cat([torch.ones(index[i]+1), torch.zeros(x.shape[1] - (index[i]+1) )])
                v2 = torch.cat([torch.zeros(index[i]+1), torch.ones(x.shape[1] - (index[i]+1)-pad_idx[i]), torch.zeros(pad_idx[i])])            
                v3 = v1 + v2
            
                m1 = torch.outer(v1, v1)
                m2 = torch.outer(v2, v2)            
                m3 = m1 + m2
                
                V.append(v3)
                M.append(m3)
            M = torch.stack(M).to(torch.bool).to(torch.device('cuda'))  
            V = torch.stack(V).to(torch.bool).to(torch.device('cuda'))  
        
#         print(M)

        M = M.unsqueeze(1)
        M = M.repeat(1, 8, 1,  1)
        M = M.view(-1, M.shape[-2], M.shape[-2])
        
#         print(M[:8, :, :].bool().logical_not())
#         print(M)
        x = self.transformer_encoder(x, mask=M.logical_not(), src_key_padding_mask=None) #V.logical_not())
#         print(np.isnan(np.array(x.detach())).all())
#         print(np.isnan(np.array(x.detach())).any())
        out1 = []
        out2 = []
        for i in range(self.bs):
            out1.append(x[i, 0, :])
            out2.append(x[i, index[i]+1, :])
            
        out1 = torch.stack(out1)
        out2 = torch.stack(out2)

        x  = torch.stack([out1, out2], axis=1) 
        # print(x.shape)
        return x
    
class BaseModel(nn.Module):
    def __init__(self, batch_size):
        super(BaseModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.pos = PositionalEncoding(256, max_len=3000) # sequence length
        self.cls1 = nn.Parameter(torch.rand(1, 256)) # 1, enc_dim
        self.cls2 = nn.Parameter(torch.rand(1, 256))  
        self.bs = batch_size

    def forward(self, x, mask, indices):
        index = indices[0]
        pad_idx = indices[1]
        
        x = self.pos(x)
        
        M = torch.ones((8 * len(x),x.shape[-2],x.shape[-2])).to(torch.bool).to(torch.device('mps')) # Code to create n(=batch size = 4) masks

        mask = mask.bool()

        x = self.transformer_encoder(x, mask = M.logical_not(), src_key_padding_mask = mask.logical_not())
        # x = self.transformer_encoder(x)
        
        out1 = []
        out2 = []
        
        for i in range(len(x)):
            patch1 = torch.mean(x[i, :index[i], :],axis=-2)
            # patch2 = torch.mean(x[i, index[i]: , :],axis=-2)
            if pad_idx[i] == 0:
                patch2 = torch.mean(x[i, index[i]: , :],axis=-2)
            else:
                patch2 = torch.mean(x[i, index[i]: -pad_idx[i], :],axis=-2)

            out1.append(patch1)
            out2.append(patch2)
            
        out1 = torch.stack(out1)
        out2 = torch.stack(out2)

        x  = torch.stack([out1, out2], axis=1) 
        return x
    
class ContrastiveLoss_test(torch.nn.Module):
    def __init__(self, margin=3.0):
        super(ContrastiveLoss_test, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive



