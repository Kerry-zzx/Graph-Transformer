import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
class GraphTransformer(nn.Module):

    def __init__(self, layers, embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout=True):
        # graph_layers-4, embed_dim-512, ff_embed_dim-1024, num_heads-8, dropout-0.2
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        # layers-4
        for _ in range(layers):
            # Todo 加残差连接等改进
            # embed_dim-512, ff_embed_dim-1024, num_heads-8, dropout-0.2, weights_dropout=True
            self.layers.append(GraphTransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout))
    
    def forward(self, x, relation, kv = None,
                self_padding_mask = None, self_attn_mask = None):
        # 图编码
        # concept_repr-token-[seq_len, bsz, 512]
        # relation-[n_max, n_max, 数据长度, 512]
        # kv-None, concept_mask-[seq_len, bsz, -1](id为pad的位置为1， 其余位置为0)
        # self_attn_mask-None
        for idx, layer in enumerate(self.layers):
            # 经过4层图Transformer
            # x-[seq_len, bsz, 512], None
            x, _ = layer(x, relation, kv, self_padding_mask, self_attn_mask)
        return x

    def get_attn_weights(self, x, relation, kv = None,
                self_padding_mask = None, self_attn_mask = None):
        attns = []
        for idx, layer in enumerate(self.layers):
            x, attn = layer(x, relation, kv, self_padding_mask, self_attn_mask, need_weights=True)
            attns.append(attn)
        attn = torch.stack(attns)
        return attn

class GraphTransformerLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformerLayer, self).__init__()
        # embed_dim-512, ff_embed_dim-1024, num_heads-8, dropout-0.2, weights_dropout=True
        # embed_dim-512, num_heads-8, dropout-0.2, weights_dropout=True
        self.self_attn = RelationMultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        # fc1-Linear[512, 1024]
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        # fc2-Linear[1024, 512]
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        # attn_layer_norm-LayerNorm(512)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        # ff_layer_norm-LayerNorm(512)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        # dropout-0.2
        self.dropout = dropout
        # 初始化全连接层参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, relation, kv = None,
                self_padding_mask = None, self_attn_mask = None,
                need_weights = False):
        # x-[seq_len, bsz, 512], relation-[n_max, n_max, 数据长度, 512]
        # kv-None, self_padding_mask-[seq_len, bsz, -1](id为pad的位置为1， 其余位置为0)
        # self_attn_mask-None, need_weights-False
        residual = x
        if kv is None:
            # query, key, value-[seq_len, bsz, 512], relation-[tgt_len, src_len, bsz, 512]
            # key_padding_maskk-[seq_len, bsz, -1](id为pad的位置为1， 其余位置为0)
            # attn_mask-None, need_weights-False

            # 使用query对relation进行查询, 获取注意力权重
            # x-[seq_len, bsz, 512]
            # self_attn-None
            x, self_attn = self.self_attn(query=x, key=x, value=x, relation=relation, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights=need_weights)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, relation=relation, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights=need_weights)

        x = F.dropout(x, p=self.dropout, training=self.training)
        # 残差连接并经过LayerNorm(512)
        x = self.attn_layer_norm(residual + x)

        residual = x
        # fc1-Linear[512, 1024]
        # x-[seq_len, bsz, 1024]
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # fc2-Linear[1024, 512]
        # x-[seq_len, bsz, 512]
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        # x-[seq_len, bsz, 512], None
        return x, self_attn

class RelationMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(RelationMultiheadAttention, self).__init__()
        # embed_dim-512, num_heads-8, dropout-0.2, weights_dropout=True
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        # head_dim-512/8-64
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # scaling-0.125
        self.scaling = self.head_dim ** -0.5

        # in_proj_weight-[512*3, 512]
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        # in_proj_bias-[3*512]
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        # relation_in_proj-[512, 512*2]
        self.relation_in_proj = nn.Linear(embed_dim, 2*embed_dim, bias=False)

        # out_proj-Linear[512, 512]
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        # weights_dropout-True
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.relation_in_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, relation, key_padding_mask=None, attn_mask=None, need_weights=False):
        """ Input shape: Time x Batch x Channel
            relation:  tgt_len x src_len x bsz x dim
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        # query, key, value-[seq_len, bsz, 512], relation-[tgt_len, src_len, bsz, 512]
        # key_padding_maskk-[seq_len, bsz](id为pad的位置为1， 其余位置为0)
         # attn_mask-None, need_weights-False
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        # query, key, value-[tgt_len-src_len, bsz, embed_dim-512]
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
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

        # q-[tgt_len, 8*bsz, 64]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim)
        # k-[src_len, 8*bsz, 64]
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)
        # v-[src_len, 8*bsz, 64]
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)

        # relation-[tgt_len, src_len, bsz, 512]
        # ra-[tgt_len, src_len, bsz, 512]
        # rb-[tgt_len, src_len, bsz, 512]
        ra, rb = self.relation_in_proj(relation).chunk(2, dim=-1)
        # ra-[tgt_len, src_len, bsz, 512]-[tgt_len, src_len, 8*bsz, 64]-
        # [src_len, tgt_len, 8*bsz, 64]
        ra = ra.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # rb-[tgt_len, src_len, bsz, 512]-[tgt_len, src_len, 8*bsz, 64]-
        # [src_len, tgt_len, 8*bsz, 64]
        rb = rb.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Todo双线性
        # q-[tgt_len, 1, 8*bsz, 64]-[tgt_len, src_len, 8*bsz, 64] 
        q = q.unsqueeze(1) + ra
        # k-[1, src_len, 8*bsz, 64]-[tgt_len, src_len, 8*bsz, 64]
        k = k.unsqueeze(0) + rb
        # scaling-0.125
        q *= self.scaling
        # q: tgt_len x src_len x bsz*heads x dim
        # k: tgt_len x src_len x bsz*heads x dim
        # v: src_len x bsz*heads x dim

        # q-[tgt_len, 1, 8*bsz, 64]-[tgt_len, src_len, 8*bsz, 64] 
        # k-[1, src_len, 8*bsz, 64]-[tgt_len, src_len, 8*bsz, 64]
        # attn_weights-[tgt_len, src_len, 8*bsz]
        attn_weights = torch.einsum('ijbn,ijbn->ijb', [q, k])
        assert list(attn_weights.size()) == [tgt_len, src_len, bsz * self.num_heads]

        if attn_mask is not None:
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(-1),
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            # key_padding_maskk-[seq_len, bsz](id为pad的位置为1， 其余位置为0)
            # attn_weights-[tgt_len, src_len, bsz, 8]
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
            # key_padding_maskk-[seq_len, bsz](id为pad的位置为1， 其余位置为0)
            # -[1, seq_len, bsz, 1]
            # 将pad位置设置为'-inf'
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(0).unsqueeze(-1),
                float('-inf')
            )
            # attn_weights-[tgt_len, src_len, 8*bsz]
            attn_weights = attn_weights.view(tgt_len, src_len, bsz * self.num_heads)


        # 求注意力分数
        # attn_weights-[tgt_len, src_len, 8*bsz]
        attn_weights = F.softmax(attn_weights, dim=1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights-[tgt_len, src_len, 8*bsz]
        # v-[src_len, 8*bsz, 64]
        # attn-[8*bsz, tgt_len, 64]
        attn = torch.einsum('ijb,jbn->bin', [attn_weights, v])
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # attn-[8*bsz, tgt_len, 64]-[tgt_len, 8*bsz, 64]-[tgt_len, bsz, 512]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # out_proj-Linear[512, 512]
        # attn-[tgt_len, bsz, 512]
        attn = self.out_proj(attn)

        if need_weights:
            # maximum attention weight over heads 
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
        else:
            attn_weights = None

        # attn-[tgt_len, bsz, 512]
        # attn_weights-None
        return attn, attn_weights

    def in_proj_qkv(self, query):
        # query, key, value-[tgt_len-src_len, bsz, embed_dim-512]
        # [tgt_len/src_len, bsz, 3*embed_dim-512]
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
        # query, key, value-[tgt_len/src_len, bsz, embed_dim-512]
        # in_proj_weight-[512*3, 512]-[out_feature, in_feature]
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        # [tgt_len/src_len, bsz, 3*embed_dim-512]
        return F.linear(input, weight, bias)

        return output