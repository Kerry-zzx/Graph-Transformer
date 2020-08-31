import torch
from torch import nn
import torch.nn.functional as F
import math

from encoder import TokenEncoder, RelationEncoder
from decoder import DecodeLayer
from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask
from graph_transformer import GraphTransformer
from data import ListsToTensor, ListsofStringToTensor, STR
from search import Hypothesis, Beam, search_by_batch

class Generator(nn.Module):
    def __init__(self, vocabs, 
                word_char_dim, word_dim,
                concept_char_dim, concept_dim,
                cnn_filters, char2word_dim, char2concept_dim,
                rel_dim, rnn_hidden_size, rnn_num_layers,
                embed_dim, ff_embed_dim, num_heads, dropout,
                snt_layers, graph_layers, inference_layers,
                pretrained_file, device):
        # 词典vocabs, token_char_dim-32, token_dim-300,
        # concept_char_dim-32, concept_dim-300, cnn_filters-[(3, 256)] 
        # char2word_dim-128, char2concept_dim-128, char2concept_dim-128,
        # rel_dim-100, rnn_hidden_size-256, rnn_num_layers-2, embed_dim-512,
        # ff_embed_dim-1024, num_heads-8, dropout-0.2, snt_layers-1,
        # graph_layers-4, inference_layers-3, pretrained_file-预训练路径,
        # device- cuda
        super(Generator, self).__init__()
        # 词典vocabs
        self.vocabs = vocabs
        # concept_encoder-TokenEncoder
        # vocabs['concept'], vocabs['concept_char'], concept_char_dim-32,
        # concept_dim-300, embed_dim-512, cnn_filters-[(3, 256)] 
        # char2concept_dim-128, dropout-0.2, pretrained_file-预训练路径
        self.concept_encoder = TokenEncoder(vocabs['concept'], vocabs['concept_char'],
                                          concept_char_dim, concept_dim, embed_dim,
                                          cnn_filters, char2concept_dim, dropout, pretrained_file)
        # relation_encoder-RelationEncoder
        # vocabs['relation']. rel_dim-100, embed_dim-512, rnn_hidden_size-256,
        # rnn_num_layers-2, dropout-0.2
        self.relation_encoder = RelationEncoder(vocabs['relation'], rel_dim, embed_dim, rnn_hidden_size, rnn_num_layers, dropout)
        # token_encoder-TokenEncoder
        # vocabs['token'], vocabs['token_char'], token_char_dim-32, token_dim-300, embed_dim-512,
        # cnn_filters-[(3, 256)], char2word_dim-128, dropout-0.2, pretrained_file-预训练路径
        self.token_encoder = TokenEncoder(vocabs['token'], vocabs['token_char'],
                        word_char_dim, word_dim, embed_dim,
                        cnn_filters, char2word_dim, dropout, pretrained_file)

        # graph_encoder-GraphTransformer
        # graph_layers-4, embed_dim-512, ff_embed_dim-1024, num_heads-8, dropout-0.2
        self.graph_encoder = GraphTransformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout)
        # snt_encoder-Transformer
        # snt_layers-1, embed_dim-512, ff_embed_dim-1024, num_heads-8, 
        # num_heads-8, dropout-0.2, with_external=True
        self.snt_encoder = Transformer(snt_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)

        # embed_dim-512
        self.embed_dim = embed_dim
        # embed_scal-sqrt(512)
        self.embed_scale = math.sqrt(embed_dim)
        # embed_dim-512, device- cuda
        self.token_position = SinusoidalPositionalEmbedding(embed_dim, device)
        # concept_depth-Embedding[32, 512]
        self.concept_depth = nn.Embedding(32, embed_dim)
        # LayerNorm
        self.token_embed_layer_norm = nn.LayerNorm(embed_dim)
        # LayerNorm
        self.concept_embed_layer_norm = nn.LayerNorm(embed_dim)
        # device- cuda
        self.self_attn_mask = SelfAttentionMask(device)
        # 词典vocabs, inference_layers-3, embed_dim-512, ff_embed_dim-1024, num_heads-8, 
        # concept_dim-300, rel_dim-100, dropout-0.2
        self.decoder = DecodeLayer(vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, concept_dim, rel_dim, dropout)
        # dropout-0.2
        self.dropout = dropout
        # probe_generator-Linear[512, 512]
        self.probe_generator = nn.Linear(embed_dim, embed_dim)
        # device- cuda
        self.device = device
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # probe_generator-Linear[512, 512]
        # 权值正态分布, 偏置0
        nn.init.normal_(self.probe_generator.weight, std=0.02)
        nn.init.constant_(self.probe_generator.bias, 0.)
        # concept_depth-Embedding[32, 512]
        nn.init.constant_(self.concept_depth.weight, 0.)

    def encoder_attn(self, inp):
        with torch.no_grad():
            # concept_encoder输出：token-[seq_len, bsz, 512]
            # embed_scal-sqrt(512)
            # concept_depth-Embedding[32, 512]
            # concept_repr-token-[seq_len, bsz, 512]
            concept_repr = self.embed_scale * self.concept_encoder(inp['concept'], inp['concept_char']) + self.concept_depth(inp['concept_depth'])
            concept_repr = self.concept_embed_layer_norm(concept_repr)
            concept_mask = torch.eq(inp['concept'], self.vocabs['concept'].padding_idx)

            # 关系编码
            # relation-[batch_size, 512]
            relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])
            # relation-[0, 512]---> 0
            relation[0,:] = 0.
            relation = relation[inp['relation']]
            sum_relation = relation.sum(dim=3)
            num_valid_paths = inp['relation'].ne(0).sum(dim=3).clamp_(min=1)
            divisor = (num_valid_paths).unsqueeze(-1).type_as(sum_relation)
            relation = sum_relation / divisor

            attn = self.graph_encoder.get_attn_weights(concept_repr, relation, self_padding_mask=concept_mask)
            # nlayers x tgt_len x src_len x  bsz x num_heads
        return attn

    def encode_step(self, inp, train=True):
        # data
        # concept-[max_len, 1]  ([id('<CLS>'), id(data), id('<PAD>')..]) for vocabs['concept']
        # concept_char-[max_len, 1] ['<CLS>'+'<STR>'+'data[i]['concept']+ '<END>+ '<PAD>'*n] for vocabs['concept_char']
        # concept_depth-[0,'data[i]['depth']+0,...]
        # relation-[n_max, n_max, 数据长度]
        # relation_bank-[len(all_relations)), n_max]顺序id
        # relation_length-长度 [len(all_relations))]
        # local_idx2token-x['idx2token']
        # local_token2idx-x['idx2token']
        # token_in-[batch_size, n_max]
        # token_char_in-[batch_size, n_max]
        # token_out-使用vocabs['predictable_token']和local_token2idx对token进行编号
        # cp_seq-['cp_seq']
        # abstract-abstract-[batch_size, n_max]
    
        # concept_encoder输出：token-[seq_len, bsz, 512]
        # embed_scal-sqrt(512)
        # concept_depth-Embedding[32, 512]
        # concept_repr-token-[seq_len, bsz, 512]
        concept_repr = self.embed_scale * self.concept_encoder(inp['concept'], inp['concept_char']) + self.concept_depth(inp['concept_depth'])
        # concept_repr-token-[seq_len, bsz, 512]
        # Layer Normalization
        # concept_repr-token-[seq_len, bsz, 512]
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        # inp['concept']-[seq_len, bsz, -1]
        # concept_mask-[seq_len, bsz](id为pad的位置为1， 其余位置为0)
        concept_mask = torch.eq(inp['concept'], self.vocabs['concept'].padding_idx)

        # inp['relation_bank'], inp['relation_length']
        # 关系编码
        # relation-[batch_size, 512]
        relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])

        # 训练阶段
        if train:
            # 根据实际relation在relation中找对应映射
            # relation-[n_max, n_max, 数据长度, 512]
            relation = relation.index_select(0, inp['relation'].view(-1)).view(*inp['relation'].size(), -1)
        else:
            #pick = inp['relation'][:,:,:,0]
            #relation = relation.index_select(0, pick.view(-1)).view(*pick.size(), -1)
            relation[0,:] = 0.
            relation = relation[inp['relation']] # i x j x bsz x num x dim
            sum_relation = relation.sum(dim=3) # i x j x bsz x dim
            num_valid_paths = inp['relation'].ne(0).sum(dim=3).clamp_(min=1) # i x j x bsz 
            divisor = (num_valid_paths).unsqueeze(-1).type_as(sum_relation)
            relation = sum_relation / divisor

        # 图编码
        # concept_repr-token-[seq_len, bsz, 512]
        # relation-[n_max, n_max, 数据长度, 512]
        # concept_mask-[seq_len, bsz, -1](id为pad的位置为1， 其余位置为0)
        # concept_repr-[seq_len, bsz, 512]
        concept_repr = self.graph_encoder(concept_repr, relation, self_padding_mask=concept_mask)

        # probe_generator-Linear[512, 512]
        # concept_repr-[seq_len, bsz, 512]-[1, bsz, 512]
        # probe-[1, bsz, 512]
        probe = torch.tanh(self.probe_generator(concept_repr[:1]))
        # concept_repr-[seq_len-1, bsz, 512]
        concept_repr = concept_repr[1:]
        # concept_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0)
        concept_mask = concept_mask[1:]
        # concept_repr-[seq_len-1, bsz, 512] 词表示
        # concept_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示
        # probe-[1, bsz, 512]
        return concept_repr, concept_mask, probe

    def work(self, data, beam_size, max_time_step, min_time_step=1):
        with torch.no_grad():
            concept_repr, concept_mask, probe = self.encode_step(data, train=False)

            mem_dict = {'graph_state':concept_repr,
                        'graph_padding_mask':concept_mask,
                        'probe':probe,
                        'local_idx2token':data['local_idx2token'],
                        'cp_seq':data['cp_seq']}
            init_state_dict = {}
            init_hyp = Hypothesis(init_state_dict, [STR], 0.)
            bsz = concept_repr.size(1)
            beams = [ Beam(beam_size, min_time_step, max_time_step, [init_hyp], self.device) for i in range(bsz)]
            search_by_batch(self, beams, mem_dict)
        return beams


    def prepare_incremental_input(self, step_seq):
        token = ListsToTensor(step_seq, self.vocabs['token'])
        token_char = ListsofStringToTensor(step_seq, self.vocabs['token_char'])
        token, token_char = token.cuda(self.device), token_char.cuda(self.device)
        return token, token_char

    def decode_step(self, inp, state_dict, mem_dict, offset, topk): 
        step_token, step_token_char = inp
        graph_repr = mem_dict['graph_state']
        graph_padding_mask = mem_dict['graph_padding_mask']
        probe = mem_dict['probe']
        copy_seq = mem_dict['cp_seq']
        local_vocabs = mem_dict['local_idx2token']
        _, bsz, _ = graph_repr.size()

        new_state_dict = {}

        token_repr = self.embed_scale * self.token_encoder(step_token, step_token_char) + self.token_position(step_token, offset)
        token_repr = self.token_embed_layer_norm(token_repr)
        for idx, layer in enumerate(self.snt_encoder.layers):
            name_i = 'token_repr_%d'%idx
            if name_i in state_dict:
                prev_token_repr = state_dict[name_i]
                new_token_repr = torch.cat([prev_token_repr, token_repr], 0)
            else:
                new_token_repr = token_repr

            new_state_dict[name_i] = new_token_repr
            token_repr, _, _ = layer(token_repr, kv=new_token_repr, external_memories=graph_repr, external_padding_mask=graph_padding_mask)
        name = 'token_state'
        if name in state_dict:
            prev_token_state = state_dict[name]
            new_token_state = torch.cat([prev_token_state, token_repr], 0)
        else:
            new_token_state = token_repr
        new_state_dict[name] = new_token_state
        LL = self.decoder(probe, graph_repr, new_token_state, graph_padding_mask, None, None, copy_seq, work=True)


        def idx2token(idx, local_vocab):
            if idx in local_vocab:
                return local_vocab[idx]
            return self.vocabs['predictable_token'].idx2token(idx)

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1) # bsz x k

        results = []
        for s, t, local_vocab in zip(topk_scores.tolist(), topk_token.tolist(), local_vocabs):
            res = []
            for score, token in zip(s, t):
                res.append((idx2token(token, local_vocab), score))
            results.append(res)

        return new_state_dict, results

    def forward(self, data):
        # data
        # concept-[max_len, 1]  ([id('<CLS>'), id(data), id('<PAD>')..]) for vocabs['concept']
        # concept_char-[max_len, 1] ['<CLS>'+'<STR>'+'data[i]['concept']+ '<END>+ '<PAD>'*n] for vocabs['concept_char']
        # concept_depth-[0,'data[i]['depth']+0,...]
        # relation-[n_max, n_max, 数据长度]
        # relation_bank-[len(all_relations)), n_max]顺序id
        # relation_length-长度 [len(all_relations))]
        # local_idx2token-x['idx2token']
        # local_token2idx-x['idx2token']
        # token_in-[batch_size, n_max]
        # token_char_in-[batch_size, n_max]
        # token_out-使用vocabs['predictable_token']和local_token2idx对token进行编号
        # cp_seq-['cp_seq']
        # abstract-abstract-[batch_size, n_max]

        # 进行词编码, 使用图Transformer, LSTM和CNN进行编码
        # concept_repr-[seq_len-1, bsz, 512] 词表示
        # concept_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示
        # probe-[1, bsz, 512]
        concept_repr, concept_mask, probe = self.encode_step(data)
        # embed_scal-sqrt(512)
        # token_encoder输出：token-[seq_len, bsz, 512]
        # token_repr-token-[seq_len, bsz, 512]
        # 位置编码token_position-[seq_len-1, bsz, 512]
        token_repr = self.embed_scale * self.token_encoder(data['token_in'], data['token_char_in']) + self.token_position(data['token_in'])
        # token_embed_layer_norm-LayerNorm
        token_repr = self.token_embed_layer_norm(token_repr)
        token_repr = F.dropout(token_repr, p=self.dropout, training=self.training)
        # token_mask-[batch_size, seq_len]
        token_mask = torch.eq(data['token_in'], self.vocabs['token'].padding_idx)
        # 输入-batch_size
        # self_attn_mask-SelfAttentionMask
        # attn_mask-[batch_size, batch_size]上三角零矩阵[100, 100]
        attn_mask = self.self_attn_mask(data['token_in'].size(0))
        # token_repr-token-[seq_len, bsz, 512]
        # self_padding_mask-[batch_size, seq_len]
        # self_attn_mask-[batch_size, batch_size]上三角零矩阵[100, 100]
        # external_memories-[seq_len-1, bsz, 512] 词表示
        # external_padding_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示
        
        # token_repr-attn-[seq_len, bsz, 512]
        token_repr = self.snt_encoder(token_repr,
                                  self_padding_mask=token_mask, self_attn_mask=attn_mask,
                                  external_memories=concept_repr, external_padding_mask=concept_mask)

        # probe-[seq_len, bsz, 512]
        probe = probe.expand_as(token_repr) # tgt_len x bsz x embed_dim
        # probe-[seq_len, bsz, 512]
        # concept_repr-[seq_len-1, bsz, 512] 词表示
        # token_repr-attn-[seq_len, bsz, 512]
        # concept_mask-[seq_len-1, bsz](id为pad的位置为1， 其余位置为0) MASK表示
        # token_mask-[batch_size, seq_len]
        # attn_mask-[batch_size, batch_size]上三角零矩阵[100, 100]
        # cp_seq-['cp_seq']
        # token_out-使用vocabs['predictable_token']和local_token2idx对token进行编号
        return self.decoder(probe, concept_repr, token_repr, concept_mask, token_mask, attn_mask, \
         data['cp_seq'], target=data['token_out'])
