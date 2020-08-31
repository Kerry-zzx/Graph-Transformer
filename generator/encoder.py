import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from transformer import Embedding
import re

def AMREmbedding(vocab, embedding_dim, pretrained_file=None, amr=False, dump_file=None):
    # char_vocab-vocabs['concept_char']
    # char_dim-concept_char_dim-32
    if pretrained_file is None:
        # Embedding[vocab.size, embedding_dim, vocab.padding_idx(id)]
        return Embedding(vocab.size, embedding_dim, vocab.padding_idx)

    tokens_to_keep = set()
    for idx in range(vocab.size):
        token = vocab.idx2token(idx)
        # TODO: Is there a better way to do this? Currently we have a very specific 'amr' param.
        if amr:
            token = re.sub(r'-\d\d$', '', token)
        tokens_to_keep.add(token)

    embeddings = {}
 
    if dump_file is not None:
        fo = open(dump_file, 'w', encoding='utf8')

    with open(pretrained_file, encoding='utf8') as embeddings_file:
        for line in embeddings_file.readlines():    
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                continue
            token = fields[0]
            if token in tokens_to_keep:
                if dump_file is not None:
                    fo.write(line)
                vector = np.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector

    if dump_file is not None:
        fo.close()

    all_embeddings = np.asarray(list(embeddings.values()))
    print ('pretrained', all_embeddings.shape)
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    embedding_matrix = torch.FloatTensor(vocab.size, embedding_dim).normal_(embeddings_mean,
                                                                            embeddings_std)

    for i in range(vocab.size):
        token = vocab.idx2token(i)

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
        else:
            if amr:
                normalized_token = re.sub(r'-\d\d$', '', token)
                if normalized_token in embeddings:
                    embedding_matrix[i] = torch.FloatTensor(embeddings[normalized_token])
    embedding_matrix[vocab.padding_idx].fill_(0.)

    return nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

class RelationEncoder(nn.Module):
    def __init__(self, vocab, rel_dim, embed_dim, hidden_size, num_layers, dropout, bidirectional=True):
        # relation_encoder-RelationEncoder
        # vocabs['relation']. rel_dim-100, embed_dim-512, hidden_size-256,
        # num_layers-2, dropout-0.2
        super(RelationEncoder, self).__init__()
        self.vocab  = vocab
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        # rel_embed-Embedding[n, 100]
        self.rel_embed = AMREmbedding(vocab, rel_dim)
        # GRU[100, 256] /2层, 双向, 池化
        self.rnn = nn.GRU(
            input_size=rel_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout if num_layers > 1 else 0.,
            bidirectional=bidirectional
        )
        # tot_dim-512
        tot_dim = 2 * hidden_size if bidirectional else hidden_size
        self.out_proj = nn.Linear(tot_dim, embed_dim)
        # out_proj-Linear [512, 512]

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, src_tokens, src_lengths):
        # inp['relation_bank'], inp['relation_length']
        # src_tokens-[seq_len, bsz]
        seq_len, bsz = src_tokens.size()
        # sorted_src_lengths-按relation长度降序排序
        # indices-降序索引
        sorted_src_lengths, indices = torch.sort(src_lengths, descending=True)
        # 按长度排序好的sequence
        # sorted_src_tokens-[seq_len, bsz]
        sorted_src_tokens = src_tokens.index_select(1, indices)
        # rel_embed-Embedding[n, 100]
        # x-[seq_len, bsz, 100]
        x = self.rel_embed(sorted_src_tokens)
        # x-[seq_len, bsz, 100] 池化
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 变长GRU前padded_sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, sorted_src_lengths.data.tolist())
 
        if self.bidirectional:
            # state_size - 4, batch_size, 256  [4, batch_size, 256]
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size 
        # 创建[4, batch_size, 256]零值矩阵
        h0 = x.data.new(*state_size).zero_()
        # self.rnn = nn.GRU(
        #     input_size=rel_dim,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=self.dropout if num_layers > 1 else 0.,
        #     bidirectional=bidirectional
        # )
        # final_h-[4, batch_size, 256]
        _, final_h = self.rnn(packed_x, h0)

        if self.bidirectional:
            def combine_bidir(outs):
                # final_h-[4, batch_size, 256]-前向后向分开[2, 2, batch_size, 256]-两层特征放一起(2, 256)[2, batch_size, 2, 256]
                # final_h-[2, batch_size, 512]
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz, -1)
            # 合并不同层特征
            final_h = combine_bidir(final_h)

        # 恢复原有排序[2, batch_size, 512]
        _, positions = torch.sort(indices)
        final_h = final_h.index_select(1, positions) # num_layers x bsz x hidden_size
        # out_proj-Linear [512, 512]
        # 取最后一层[batch_size, 512]进行全连接
        # output-[batch_size, 512]
        output = self.out_proj(final_h[-1]) 

        return output



class TokenEncoder(nn.Module):
    def __init__(self, token_vocab, char_vocab, char_dim, token_dim, embed_dim, filters, char2token_dim, dropout, pretrained_file=None):
        # concept_encoder-TokenEncoder
        # token_vocab-vocabs['concept']
        # char_vocab-vocabs['concept_char']
        # char_dim-concept_char_dim-32
        # token_dim-concept_dim-300
        # embed_dim-512
        # filters-cnn_filters-[(3, 256)] 
        # char2token_dim-128
        # dropout-0.2
        # pretrained_file-预训练路径
        super(TokenEncoder, self).__init__()
        # char_vocab-vocabs['concept_char']
        # char_dim-concept_char_dim-32
        # nn.Embedding(n, 32)
        self.char_embed = AMREmbedding(char_vocab, char_dim)
        # token_vocab-vocabs['concept']
        # token_dim-concept_dim-300
        # pretrained_file-预训练路径 None
        # nn.Embedding(n, 300)
        self.token_embed = AMREmbedding(token_vocab, token_dim, pretrained_file)
        # filters-cnn_filters-[(3, 256)] 
        # char_dim-concept_char_dim-32
        # char2token_dim-128
        self.char2token = CNNEncoder(filters, char_dim, char2token_dim)
        # tot_dim-128+300
        tot_dim = char2token_dim + token_dim
        # out_proj-Linear[428, 512]
        self.out_proj = nn.Linear(tot_dim, embed_dim)
        # char_dim-32
        self.char_dim = char_dim
        # token_dim-concept_dim-300
        self.token_dim = token_dim
        # dropout-0.2
        self.dropout = dropout
        # 初始化out_proj权重参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, token_input, char_input):
        # data['concept'], data['concept_char']
        # seq_len-序列长度
        # bsz-batch_size大小
        seq_len, bsz, _ = char_input.size()
        # char_embed-nn.Embedding(n, 32)
        # char_repr-[seq_len * bsz, -1, 32]
        char_repr = self.char_embed(char_input.view(seq_len * bsz, -1))
        # 进行卷积编码(Conv1D), 扩大单词embedding维度32-256
        # char2token返回[seq_len * bsz, 128]
        # char_repr-[seq_len, bsz, 128]
        char_repr = self.char2token(char_repr).view(seq_len, bsz, -1)
        # token_input-data['concept']
        # token_embed-nn.Embedding(n, 300)
        # token_repr-[seq_len, bsz, 300]
        token_repr = self.token_embed(token_input)

        # token-[seq_len, bsz, 428]
        token = F.dropout(torch.cat([char_repr,token_repr], -1), p=self.dropout, training=self.training)
        # out_proj-Linear[428, 512]
        token = self.out_proj(token)
        # token-[seq_len, bsz, 512]
        return token

class CNNEncoder(nn.Module):
    def __init__(self, filters, input_dim, output_dim, highway_layers=1):
        # filters-cnn_filters-[(3, 256)] 
        # input_dim-concept_char_dim-32
        # output_dim-128
        super(CNNEncoder, self).__init__()
        self.convolutions = nn.ModuleList()
        # 3, 256
        for width, out_c in filters:
            # Conv1d-[32, 256, kernel_size=3]
            self.convolutions.append(nn.Conv1d(input_dim, out_c, kernel_size=width))
        # [(3, 256)]
        # final_dim-256
        final_dim = sum(f[1] for f in filters)
        # final_dim-256
        # highway_layers-1
        self.highway = Highway(final_dim, highway_layers)
        # out_proj-Linear [256, 128]
        self.out_proj = nn.Linear(final_dim, output_dim)
        # 初始化out_proj权值
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, input):
        # input: batch_size x seq_len x input_dim
        # char_repr-[seq_len * bsz, -1, 32]
        # x-[seq_len * bsz, 32, -1]
        # 将输入转置为了后面conv1d
        x  = input.transpose(1, 2)
        conv_result = []
        for i, conv in enumerate(self.convolutions):
            # Conv1d-[32, 256, kernel_size=3]  将词向量扩充到256
            # y-[seq_len * bsz, 256, -1]
            y = conv(x)
            # y-[seq_len * bsz, 256]
            y, _ = torch.max(y, -1)
            # ReLU激活函数
            y = F.relu(y)
            conv_result.append(y)

        # 合并卷积处理后结果 conv_result-[seq_len * bsz, 256]
        conv_result = torch.cat(conv_result, dim=-1)
        # 全连接层Linear, 权值合并
        # conv_result-[seq_len * bsz, 256]
        conv_result = self.highway(conv_result)
        # out_proj-Linear [256, 128]
        # [seq_len * bsz, 128]
        return self.out_proj(conv_result) #  batch_size x output_dim

class Highway(nn.Module):
    def __init__(self, input_dim, layers):
        super(Highway, self).__init__()
        # input_dim-256
        # layers-1
        self.input_dim = input_dim
        # layers-Linear[256, 512]
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(layers)])
        # 初始化layers权值
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)

    def forward(self, x):
        # 合并卷积处理后结果 conv_result-[seq_len * bsz, 256]
        for layer in self.layers:
            # new_x-[seq_len * bsz, 512]
            new_x = layer(x)
            # new_x-[seq_len * bsz, 256]
            new_x, gate = new_x.chunk(2, dim=-1)
            new_x = F.relu(new_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (1 - gate) * new_x
        return x
