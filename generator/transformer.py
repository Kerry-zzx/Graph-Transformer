import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class Transformer(nn.Module):

    def __init__(self, layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=False, weights_dropout=True):
        # snt_layers-1, embed_dim-512, ff_embed_dim-1024, num_heads-8, 
        # num_heads-8, dropout-0.2, with_external=True
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList()
        # layers-1层
        for _ in range(layers):
            # embed_dim-512, ff_embed_dim-1024, num_heads-8, dropout-0.2, with_external-True, weights_dropout-True
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, with_external, weights_dropout))
    
    def forward(self, x, kv = None,
                self_padding_mask = None, self_attn_mask = None,
                external_memories = None, external_padding_mask=None):
        # token_repr-token-[seq_len, bsz, 512]
        # self_padding_mask-[batch_size, seq_len]
        # self_attn_mask-[batch_size, batch_size]上三角零矩阵[100, 100]
        # external_memories-[seq_len-1, bsz, 512] 词表示
        # external_padding_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示
        for idx, layer in enumerate(self.layers):
            # x-attn-[seq_len, bsz, 512]
            # self_attn, external_attn-None
            x, _, _ = layer(x, kv, self_padding_mask, self_attn_mask, external_memories, external_padding_mask)
        return x

class TransformerLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, with_external=False, weights_dropout=True):
        super(TransformerLayer, self).__init__()
        # embed_dim-512, ff_embed_dim-1024, num_heads-8, dropout-0.2, with_external-True, 
        # weights_dropout-True
        
        # embed_dim-512, num_heads-8, dropout-0.2, weights_dropout-True
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        # fc1-Linear[512, 1024]
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        # fc2-Linear[1024, 512]
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.with_external = with_external
        self.dropout = dropout
        if self.with_external:
            self.external_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
            self.external_layer_norm = nn.LayerNorm(embed_dim)
        # 初始化全连接层参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, kv = None,
                self_padding_mask = None, self_attn_mask = None,
                external_memories = None, external_padding_mask=None,
                need_weights = False):
        # x: seq_len x bsz x embed_dim
        
        # x-[seq_len, bsz, 512]
        # self_padding_mask-[batch_size, seq_len]
        # self_attn_mask-[batch_size, batch_size]上三角零矩阵[100, 100]
        # external_memories-[seq_len-1, bsz, 512] 词表示
        # external_padding_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示
        residual = x
        if kv is None:
            # query-[seq_len, bsz, 512]
            # key-[seq_len, bsz, 512]
            # value-[seq_len, bsz, 512]
            # key_padding_mask-[batch_size, seq_len]
            # attn_mask-[batch_size, batch_size]上三角零矩阵[100, 100]
            # need_weights-False

            # x-attn-[seq_len, bsz, 512]
            x, self_attn = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights=need_weights)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights=need_weights)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        if self.with_external:
            residual = x
            x, external_attn = self.external_attn(query=x, key=external_memories, value=external_memories, key_padding_mask=external_padding_mask, need_weights=need_weights)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn = None

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # fc2-Linear[1024, 512]
        # x-attn-[seq_len, bsz, 512]
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        # x-attn-[seq_len, bsz, 512]
        # self_attn, external_attn-None
        return x, self_attn, external_attn

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        # embed_dim-512, num_heads-8, dropout-0.2, weights_dropout-True
        # decoder-alignment_layer-MultiheadAttention[embed_dim-512, 1, dropout-0.2, weights_dropout-False]
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        # head_dim-512/8-64
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # scaling-0.125
        self.scaling = self.head_dim ** -0.5

        # in_proj_weight-[3*512, 512]
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        # out_proj-Linear[512, 512]
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False):
        """ Input shape: Time x Batch x Channel
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        # query-[seq_len, bsz, 512]
        # key-[seq_len, bsz, 512]
        # value-[seq_len, bsz, 512]
        # key_padding_mask-[batch_size, seq_len]
        # attn_mask-[batch_size, batch_size]上三角零矩阵[100, 100]
        # need_weights-False
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            # query, key, value-[tgt_len-src_len, bsz, embed_dim-512]
            # query经过全连接层生成q, k, v
            # [tgt_len/src_len, bsz, embed_dim-512]
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling


        # q-[seq_len, 8*bsz, 64]-[8*bsz, seq_len, 64]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # q-[seq_len, 8*bsz, 64]-[8*bsz, seq_len, 64]
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # q-[seq_len, 8*bsz, 64]-[8*bsz, seq_len, 64]
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        # k,v: bsz*heads x src_len x dim
        # q: bsz*heads x tgt_len x dim 

        # Batch乘积
        # q-[8*bsz, seq_len, 64]
        # k-[8*bsz, 64, seq_len]
        # attn_weights-[8*bsz, seq_len, seq_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # attn_mask-[batch_size, batch_size]上三角零矩阵[100, 100]
        if attn_mask is not None:
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0),
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            # key_padding_mask-[batch_size, seq_len]
            # attn_weights-[8*bsz, seq_len, seq_len]-[bsz, 8, seq_len, seq_len]
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            # key_padding_mask-[seq_len, batch_size]-[batch_size, 1, seq_len, 1]
            attn_weights.masked_fill_(
                key_padding_mask.transpose(0, 1).unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)


        # attn_weights-[8*bsz, seq_len, seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights-[8*bsz, seq_len, seq_len]
        # v-[8*bsz, seq_len, 64]
        # 对v权值之后attn-[8*bsz, seq_len, 64]
        attn = torch.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # attn-[seq_len, bsz, 512]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # out_proj-Linear[512, 512]
        # attn-[seq_len, bsz, 512]
        attn = self.out_proj(attn)

        if need_weights:
            # maximum attention weight over heads 
            # attn_weights-[bsz, 1, tgt_len, src_len]
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

            # attn_weights-[bsz, tgt_len, src_len]
            attn_weights, _ = attn_weights.max(dim=1)
            # attn_weights-[tgt_len, bsz, src_len]
            attn_weights = attn_weights.transpose(0, 1)
        else:
            attn_weights = None

        # attn-[seq_len, bsz, 512]
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, std=0.02)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class SelfAttentionMask(nn.Module):
    def __init__(self, device, init_size=100):
        # device-'cuda'
        super(SelfAttentionMask, self).__init__()
        # weights-上三角零矩阵[100, 100]
        self.weights = SelfAttentionMask.get_mask(init_size)
        self.device = device

    @staticmethod
    def get_mask(size):
        # weights-[100, 100]-上三角零矩阵[100, 100]
        weights = torch.ones((size, size), dtype = torch.uint8).triu_(1)
        return weights

    def forward(self, size):
        # 输入-batch_size
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        # res-[batch_size, batch_size]上三角零矩阵[100, 100]
        res = self.weights[:size,:size].detach().to(self.device).detach()
        return res

class LearnedPositionalEmbedding(nn.Module):
    """This module produces LearnedPositionalEmbedding.
    """
    def __init__(self, embedding_dim, device, max_size=512):
        super(LearnedPositionalEmbedding, self).__init__()
        self.weights = nn.Embedding(max_size, embedding_dim)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weights.weight, 0.)

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        seq_len, bsz = input.size()
        positions = (offset + torch.arange(seq_len)).to(self.device)
        res = self.weights(positions).unsqueeze(1).expand(-1, bsz, -1)
        return res

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    """
    def __init__(self, embedding_dim, device, init_size=512):
        # embed_dim-512, device- cuda
        super(SinusoidalPositionalEmbedding, self).__init__()
        # embedding_dim-512
        self.embedding_dim = embedding_dim
        # 位置编码: weights-[512, 512]
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim
        )
        self.device = device

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        # num_embeddings-512, embedding_dim-512
        # half_dim-256
        half_dim = embedding_dim // 2
        # emb-log(10000)/255
        emb = math.log(10000) / (half_dim - 1)
        # emb-exp([0, 1, ..., 255]*(-0.036))
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        # 位置编码
        # emb-[512, 256]
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        # 位置编码: emb-[512, 512]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, input, offset=0):
        """Input is expected to be of size [seq_len x bsz]."""
        # token_in-[seq_len, batch_size]
        seq_len, bsz = input.size()
        mx_position = seq_len + offset
        if self.weights is None or mx_position > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                mx_position,
                self.embedding_dim,
            )

        # 位置编号-positions-[0, 1, ..., seq_len-1]
        positions = offset + torch.arange(seq_len)
        # res-[seq_len-1, 512]-[seq_len-1, 1, 512]-[seq_len-1, bsz, 512]
        res = self.weights.index_select(0, positions).unsqueeze(1).expand(-1, bsz, -1).detach().to(self.device).detach()
        return res
