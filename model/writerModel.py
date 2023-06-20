import torch
from torch import nn
import numpy as np
import math

from einops import rearrange
from einops.layers.torch import Rearrange

from sklearn.metrics.pairwise import euclidean_distances

# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)

# def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
#     _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

#     y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
#     assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
#     omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
#     omega = 1. / (temperature ** omega)

#     y = y.flatten()[:, None] * omega[None, :]
#     x = x.flatten()[:, None] * omega[None, :] 
#     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
#     return pe.type(dtype)

# # classes

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )
#     def forward(self, x):
#         return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.norm = nn.LayerNorm(dim)

#         self.attend = nn.Softmax(dim = -1)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         self.to_out = nn.Linear(inner_dim, dim, bias = False)

#     def forward(self, x, mask):
#         mask1 = mask.view(mask.shape[0], 1, -1, 1)
#         mask2 = mask.view(mask.shape[0], 1, 1, -1)
#         x = self.norm(x)

#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         dots = dots * mask1 * mask2
#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Attention(dim, heads = heads, dim_head = dim_head),
#                 FeedForward(dim, mlp_dim)
#             ]))
#     def forward(self, x, mask):
#         for attn, ff in self.layers:
#             x = attn(x, mask) + x
#             x = ff(x) + x
#         return x

# class writerViT(nn.Module):
#     def __init__(self, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
#         super().__init__()

#         patch_height, patch_width = pair(patch_size)

#         #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         #num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width

#         self.to_patch_embedding = nn.Sequential(
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

#         self.to_latent = nn.Identity()
#         self.linear_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#         self.relu = nn.ReLU()

#     def forward(self, img, mask):
#         *_, h, w, dtype = *img.shape, img.dtype
#         #print("-----inside forward-----")
#         #print(img.shape)
#         x = self.to_patch_embedding(img)

#         pe = posemb_sincos_2d(x)
        
#         x = rearrange(x, 'b ... d -> b (...) d') + pe
#         #print(x.shape)        

#         x = self.transformer(x, mask)
        
#         x = x.mean(dim = 1)
#         x = self.to_latent(x)
#         x = self.linear_head(x)
#         #print('before relu :', x[:, 0])
#         #print('------------')
#         return self.relu(x)

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
    def __init__(self):
        super(Model, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.pos = PositionalEncoding(256, max_len=3000) # sequence length
        self.cls1 = nn.Parameter(torch.rand(1, 256)) # 1, enc_dim
        self.cls2 = nn.Parameter(torch.rand(1, 256))  

    def forward(self, x, mask, indices):
#         x = torch.cat([self.cls1.repeat(4, 1).unsqueeze(1), x[:, :4, :], self.cls2.repeat(4, 1).unsqueeze(1),x[:,4:,:]], axis=1)
        # index = [3, 4, 2, 2]
        # pad_idx = [2,1,2,4]
        index = indices[0]
        pad_idx = indices[1]
        new_x = []
        
        # To insert class tokens
        for i in range(4):
            new_x.append(torch.cat([self.cls1.unsqueeze(1), x[i, :index[i], :].unsqueeze(0), self.cls2.unsqueeze(1), x[i, index[i]: , :].unsqueeze(0)], axis=1))
            
            
        x = torch.cat(new_x)
        x = self.pos(x)
        
#         M = torch.ones((4,9,9)).to(torch.float32)  # Code to create n(=batch size = 4) masks
        M = []
        V = []
        with torch.no_grad():
            for i in range(4):
                v1 = torch.cat([torch.ones(index[i]+1), torch.zeros(x.shape[1] - (index[i]+1) )])
                v2 = torch.cat([torch.zeros(index[i]+1), torch.ones(x.shape[1] - (index[i]+1)-pad_idx[i]), torch.zeros(pad_idx[i])])            
                v3 = v1 + v2
            
                m1 = torch.outer(v1, v1)
                m2 = torch.outer(v2, v2)            
                m3 = m1 + m2
                
                V.append(v3)
                M.append(m3)
            M = torch.stack(M).to(torch.bool)
            V = torch.stack(V).to(torch.bool).to(torch.device('cuda'))  
        
#         print(M)

        M = M.unsqueeze(1)
        M = M.repeat(1, 8, 1,  1)
        M = M.view(-1, M.shape[-2], M.shape[-2]).to(torch.device('cuda'))  
        
#         print(M[:8, :, :].bool().logical_not())
#         print(M)

        x = self.transformer_encoder(x, mask=M.logical_not(), src_key_padding_mask=None) #V.logical_not())
#         print(np.isnan(np.array(x.detach())).all())
#         print(np.isnan(np.array(x.detach())).any())
        out1 = []
        out2 = []
        for i in range(4):
            out1.append(x[i, 0, :])
            out2.append(x[i, index[i]+1, :])
            
        out1 = torch.stack(out1)
        out2 = torch.stack(out2)

        x  = torch.stack([out1, out2], axis=1) 
#         print(x.shape)
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
        
        M = torch.ones((8 * self.bs,x.shape[-2],x.shape[-2])).to(torch.bool).to(torch.device('cuda')) # Code to create n(=batch size = 4) masks

        mask = mask.bool()

        x = self.transformer_encoder(x, mask = M.logical_not(), src_key_padding_mask = mask.logical_not())
        # x = self.transformer_encoder(x)
        
        out1 = []
        out2 = []
        
        for i in range(self.bs):
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
    
class ContrastiveLoss(nn.Module):
    def __init__(self, alpha, beta, margin):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, pred, y):
        # D = torch.sum(torch.sqrt((pred[:, 0, :] - pred[:, 1, :])**2), axis=-1)
        D = (pred[:, 0, :] - pred[:, 1, :]).pow(2).sum(1).sqrt()

        term1 = (1 - y) * (D**2)

        term2 = y * torch.max(torch.zeros_like(D), (self.margin - D))**2
        loss =  self.alpha * term1 + self.beta * term2
               
        return torch.mean(loss, dtype=torch.float32)

