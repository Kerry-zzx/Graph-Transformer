import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils import compute_f_by_tensor
from transformer import MultiheadAttention, Transformer

from utils import label_smoothed_nll_loss

class TokenGenerator(nn.Module):
    def __init__(self, vocabs, embed_dim, token_size, dropout):
        super(TokenGenerator, self).__init__()
        # 词典vocabs, embed_dim-512, token_size-300, dropout-0.2
        # alignment_layer-MultiheadAttention[embed_dim-512, num-head-1, dropout-0.2, weights_dropout-False]
        self.alignment_layer = MultiheadAttention(embed_dim, 1, dropout, weights_dropout=False)
        self.alignment_layer_norm = nn.LayerNorm(embed_dim)
        # transfer-Linear[512, 300]
        self.transfer = nn.Linear(embed_dim, token_size)
        # generator-Linear[300, vocabs['predictable_token'].size]
        self.generator = nn.Linear(token_size, vocabs['predictable_token'].size)
        # diverter-Linear[300, 2]
        self.diverter = nn.Linear(token_size, 2)
        self.vocabs = vocabs
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.transfer.weight, std=0.02)
        nn.init.normal_(self.diverter.weight, std=0.02)
        nn.init.normal_(self.generator.weight, std=0.02) 
        nn.init.constant_(self.diverter.bias, 0.)
        nn.init.constant_(self.transfer.bias, 0.)
        nn.init.constant_(self.generator.bias, 0.)

    def forward(self, outs, graph_state, graph_padding_mask, copy_seq,
                target=None, work=False):
        # outs-[seq_len, bsz, 512]
        # graph_state-concept_repr-[seq_len-1, bsz, 512] 词表示
        # graph_padding_mask-concept_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示
        # copy_seq-['cp_seq']
        # target-token_out-使用vocabs['predictable_token']和local_token2idx对token进行编号
        # work-False

        # 使用outs作为query, graph_state作为key和value, 获取权值加和后的graph_state
        # x-[tgt_len, bsz, 512] 
        # attn_weights-[tgt_len, bsz, src_len]
        x, alignment_weight = self.alignment_layer(outs, graph_state, graph_state,
                                                    key_padding_mask=graph_padding_mask,
                                                    need_weights=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 残差后Layer Normalization
        outs = self.alignment_layer_norm(outs + x)

        seq_len, bsz, _ = outs.size()
        # outs-[seq_len, bsz, 512]
        # transfer-Linear[512, 300]
        # outs_token-[seq_len, bsz, 300]
        outs_token = torch.tanh(self.transfer(outs))
        outs_token = F.dropout(outs_token, p=self.dropout, training=self.training)

        # TODO: Copy 机制
        # diverter-Linear[300, 2]
        # outs_token-[seq_len, bsz, 300]-[seq_len, bsz, 2]
        # 生成门gen_gate: [seq_len, bsz, 1]
        # 复制门copy_gate: [seq_len, bsz, 1]
        gen_gate, copy_gate = F.softmax(self.diverter(outs_token), -1).chunk(2, dim=-1)
        
        # TODO: LSTM
        # generator-Linear[300, vocabs['predictable_token'].size]
        # outs_token-[seq_len, bsz, 300]
        # probs-[seq_len, bsz, vocabs['predictable_token'].size]  预测词汇概率
        probs = gen_gate * F.softmax(self.generator(outs_token), -1)

        # copy_seq-['cp_seq']
        # tot_ext-copy_seq中词汇编号最大值
        tot_ext = 1 + copy_seq.max().item()
        # 词汇大小vocab_size-vocabs['predictable_token'].size
        vocab_size = probs.size(-1)

        # 如果超过vocab_size
        if tot_ext - vocab_size > 0:
            ext_probs = probs.new_zeros((1, 1, tot_ext - vocab_size)).expand(seq_len, bsz, -1)
            probs = torch.cat([probs, ext_probs], -1)

        # copy_seq-[bsz, seq_len]
        # index-[tgt_len, bsz, src_len]
        index = copy_seq.transpose(0, 1).contiguous().view(1, bsz, -1).expand(seq_len, -1, -1)
        
        # 复制门copy_gate-[seq_len, bsz, src_len]
        # alignment_weight-[tgt_len, bsz, src_len]
        # copy_probs-复制门概率-[tgt_len, bsz, src_len]
        copy_probs = (copy_gate * alignment_weight).view(seq_len, bsz, -1)
        # probs-[seq_len, bsz, vocabs['predictable_token'].size]  预测词汇概率
        # ?????
        # probs-[seq_len, bsz, vocabs['predictable_token'].size]  预测词汇概率
        probs = probs.scatter_add_(-1, index, copy_probs)
        ll = torch.log(probs + 1e-12)
        
        if work:
            return ll

        # target-token_out-使用vocabs['predictable_token']和local_token2idx对token进行编号
        # target-根据target选取最后一维
        token_loss = -ll.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        token_mask = torch.eq(target, self.vocabs['predictable_token'].padding_idx)
        token_loss = token_loss.masked_fill_(token_mask, 0.).sum(0)
        return token_loss

class DecodeLayer(nn.Module):

    def __init__(self, vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, token_size, rel_size, dropout):
        # 词典vocabs, inference_layers-3, embed_dim-512, ff_embed_dim-1024, num_heads-8, 
        # token_size-300, rel_size-100, dropout-0.2
        super(DecodeLayer, self).__init__()
        # inference_layers-3,  embed_dim-512, ff_embed_dim-1024, num_heads-8, dropout-0.2, with_external-True
        self.inference_core = Transformer(inference_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        # 词典vocabs, embed_dim-512, token_size-300, dropout-0.2
        self.token_generator = TokenGenerator(vocabs, embed_dim, token_size, dropout)
        self.dropout = dropout
        self.vocabs = vocabs

    def forward(self, probe, graph_state, snt_state,
                graph_padding_mask, snt_padding_mask, attn_mask,
                copy_seq, target=None, work=False):
        # probe: tgt_len x bsz x embed_dim
        # snt_state, graph_state: seq_len x bsz x embed_dim

        # probe-[seq_len, bsz, 512]
        # graph_state-concept_repr-[seq_len-1, bsz, 512] 词表示
        # snt_state-token_repr-attn-[seq_len, bsz, 512]
        # graph_padding_mask-concept_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示
        # snt_padding_mask-token_mask-[batch_size, seq_len]
        # attn_mask-[batch_size, batch_size]上三角零矩阵[100, 100]
        # copy_seq-cp_seq-['cp_seq']
        # target-token_out-使用vocabs['predictable_token']和local_token2idx对token进行编号
        # work-False
        
        # outs-[seq_len, bsz, 512]
        outs = F.dropout(probe, p=self.dropout, training=self.training)
        # outs-[seq_len, bsz, 512], kv-snt_state-[seq_len, bsz, 512]
        # self_padding_mask-snt_padding_mask-token_mask-[batch_size, seq_len]
        # external_memories-graph_state-[seq_len-1, bsz, 512] 词表示
        # external_padding_mask-graph_padding_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示

        # 使用outs作为query, snt_state作为key和value, 获得outs,
        # 继续使用graph_state作为key和value获得最终输出[seq_len, bsz, 512]

        # outs-[seq_len, bsz, 512]
        outs = self.inference_core(outs, kv=snt_state,
                    self_padding_mask=snt_padding_mask, self_attn_mask=attn_mask,
                    external_memories=graph_state, external_padding_mask=graph_padding_mask)

        if work:
            concept_ll = self.token_generator(outs, graph_state, graph_padding_mask, copy_seq, work=True)
            return concept_ll

        # outs-[seq_len, bsz, 512]
        # graph_state-concept_repr-[seq_len-1, bsz, 512] 词表示
        # graph_padding_mask-concept_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示
        # copy_seq-cp_seq-['cp_seq']
        # target-token_out-使用vocabs['predictable_token']和local_token2idx对token进行编号
        # work-False

        # 生成token的损失值
        token_loss = self.token_generator(outs, graph_state, graph_padding_mask, copy_seq, target=target, work=False)
        token_tot = snt_padding_mask.size(0) - snt_padding_mask.float().sum(0)
        token_loss = token_loss / token_tot
        return token_loss.mean()
