import random
import torch
from torch import nn
import numpy as np
import json

PAD, UNK = '<PAD>', '<UNK>'
CLS = '<CLS>'
STR, END = '<STR>', '<END>'
SEL, rCLS, TL = '<SELF>', '<rCLS>', '<TL>'

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        # 文件路径, 最小出现数目min_occur_cnt, 特殊token
        # vocabs['concept'] = Vocab(args.concept_vocab, 5, [CLS])
        # vocabs['token'] = Vocab(args.token_vocab, 5, [STR, END])
        # vocabs['predictable_token'] = Vocab(args.predictable_token_vocab, 5, [END])
        # vocabs['token_char'] = Vocab(args.token_char_vocab, 100, [STR, END])
        # vocabs['concept_char'] = Vocab(args.concept_char_vocab, 100, [STR, END])
        # vocabs['relation'] = Vocab(args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
        # 特殊token
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._priority = dict()
        num_tot_tokens = 0
        num_vocab_tokens = 0
        for line in open(filename).readlines():
            try:
                token, cnt = line.strip().split('\t')
                cnt = int(cnt)
                # token总数目
                num_tot_tokens += cnt
            except:
                print(line)
            if cnt >= min_occur_cnt:
                # 将token添加到idx2token
                idx2token.append(token)
                # 符合要求token数目
                num_vocab_tokens += cnt
            # _priority: {token: cnt}
            self._priority[token] = int(cnt)
        # coverage: 正样本比例
        self.coverage = num_vocab_tokens/num_tot_tokens
        # ._token2idx: {token: id}
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        # [token]
        self._idx2token = idx2token
        # PAD的id
        self._padding_idx = self._token2idx[PAD]
        # UNK的id
        self._unk_idx = self._token2idx[UNK]

    def priority(self, x):
        # 获取x在vocab中id
        return self._priority.get(x, 0)

    @property
    def size(self):
        # vocab长度
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

def _back_to_txt_for_check(tensor, vocab, local_idx2token=None):
    for bid, xs in enumerate(tensor.t().tolist()):
        txt = []
        for x in xs:
            if x == vocab.padding_idx:
                break
            if x >= vocab.size:
                assert local_idx2token is not None
                assert local_idx2token[bid] is not None
                tok = local_idx2token[bid][x]
            else:
                tok = vocab.idx2token(x)
            txt.append(tok)
        txt = ' '.join(txt)
        print (txt)

def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    # ['<CLS>'+data[i]['concept']], vocabs['concept'], 0
    # '<PAD>'的id
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        # i: id, x: '<CLS>'+data[i]['concept']
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        # 随机选取
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        # 返回'<CLS>'和data[i]['concept']的id
        return vocab.token2idx(w)

    # 最大长度max_len
    max_len = max(len(x) for x in xs)
    ys = []
    # 编码！！
    for i, x in enumerate(xs):
        # i: id, x: '<CLS>'+data[i]['concept']
        # toIdx: '<CLS>'和data[i]['concept']的id
        # 不足地方补充'<PAD>'id
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        # [id('<CLS>'), id(data), id('<PAD>')..]
        ys.append(y)
    # 转成张量, 转置
    # [max_len, 1]([id('<CLS>'), id(data), id('<PAD>')..])
    data = torch.LongTensor(ys).t_().contiguous()
    return data

def ListsofStringToTensor(xs, vocab, max_string_len=20):
    # ['<CLS>'+data[i]['concept']], vocabs['concept_char']
    # max_len 最大长度
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        # ['<CLS>'+data[i]['concept']+'<PAD>'*n]
        y = x + [PAD]*(max_len -len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            # ['<CLS>'+'<STR>'+'data[i]['concept']+ '<END>+ '<PAD>'*n]
            zs.append(vocab.token2idx([STR]+z+[END]) + [vocab.padding_idx]*(max_string_len - len(z)))
        ys.append(zs)

    # [max_len, 1] ['<CLS>'+'<STR>'+'data[i]['concept']+ '<END>+ '<PAD>'*n]
    data = torch.LongTensor(ys).transpose(0, 1).contiguous()
    return data

def ArraysToTensor(xs):
    "list of numpy array, each has the same demonsionality"
    # _relation_type -[len(data), [(n+1)*(n+1)]]
    # x-shape(x)-([n, n], [m, m], ...)
    x = np.array([ list(x.shape) for x in xs])
    # shape-[数据长度, n_max, n_max]
    shape = [len(xs)] + list(x.max(axis = 0))
    # [数据长度, n_max, n_max]
    data = np.zeros(shape, dtype=np.int)
    # x-[n, n]
    for i, x in enumerate(xs):
        # [n, n]
        slicing_shape = list(x.shape)
        # 取x0维第i个, 1-2维第0到n_max
        slices = tuple([slice(i, i+1)]+[slice(0, x) for x in slicing_shape])
        data[slices] = x
        tensor = torch.from_numpy(data).long()
    # tensor-[数据长度, n_max, n_max]
    return tensor

def batchify(data, vocabs, unk_rate=0., train=True):
    # batch:[data{}], 词典vocabs, unk_rate 0, train True
    # ['<CLS>'+data[i]['concept']], vocabs['concept'], 0
    # _conc-[max_len, 1]  ([id('<CLS>'), id(data), id('<PAD>')..]) for vocabs['concept']
    _conc = ListsToTensor([ [CLS]+x['concept'] for x in data], vocabs['concept'], unk_rate=unk_rate)
    # ['<CLS>'+data[i]['concept']], vocabs['concept_char']
    # _conc_char-[max_len, 1] ['<CLS>'+'<STR>'+'data[i]['concept']+ '<END>+ '<PAD>'*n] for vocabs['concept_char']
    _conc_char = ListsofStringToTensor([ [CLS]+x['concept'] for x in data], vocabs['concept_char'])
    # _depth-[0,'data[i]['depth']+0,...]
    _depth = ListsToTensor([ [0]+x['depth'] for x in data])


    # 训练阶段
    if train:

        all_relations = dict()
        # '<CLS>' id
        cls_idx = vocabs['relation'].token2idx(CLS)
        # '<rCLS>' id
        rcls_idx = vocabs['relation'].token2idx(rCLS)
        # '<SELF>' id
        self_idx = vocabs['relation'].token2idx(SEL)
        # {('<CLS>' id) : 0}
        all_relations[tuple([cls_idx])] = 0
        all_relations[tuple([rcls_idx])] = 1
        all_relations[tuple([self_idx])] = 2

        _relation_type = []
        # batch:[data{}]
        for bidx, x in enumerate(data):
            # x['concept']长度
            n = len(x['concept'])
            # [2, 0, .......]
            brs = [ [2]+[0]*(n) ]
            for i in range(n):
                rs = [1]
                for j in range(n):
                    # 遍历所有路径 [i][j]
                    all_path = x['relation'][str(i)][str(j)]
                    # 随机选取一条路径
                    path = random.choice(all_path)['edge']
                    # '<SELF>', '<rCLS>', '<TL>'
                    # 如果没有路径
                    if len(path) == 0: # self loop
                        # '<SELF>'
                        path = [SEL]
                    # 如果路径长度太长
                    if len(path) > 8: # too long distance
                        # '<TL>'
                        path = [TL]
                    # path-relation id
                    path = tuple(vocabs['relation'].token2idx(path))
                    # rtype: {path relation id: 编号}
                    # rtype: all_relations中的编号
                    rtype = all_relations.get(path, len(all_relations))
                    # all_relations{path relation id: len}
                    if rtype == len(all_relations):
                        all_relations[path] = len(all_relations)
                    # all_relations中的编号
                    rs.append(rtype)
                # 转成numpy.array
                rs = np.array(rs, dtype=np.int)
                # for node i
                # [2, 0, .......]
                # [1, relations_id1, ..., relations_idn]
                brs.append(rs)
            # brs-关系矩阵: [(n+1)*(n+1)]
            brs = np.stack(brs)
            # _relation_type -[len(data), [(n+1)*(n+1)]]
            _relation_type.append(brs)
        # _relation_type -[len(data), [(n+1)*(n+1)]]
        # 返回tensor-[数据长度, n_max, n_max]
        # _relation_type-[n_max, n_max, 数据长度]
        _relation_type = ArraysToTensor(_relation_type).transpose_(0, 2)
        # _relation_bank[_relation_type[i][j][b]] => from j to i go through what 

        # B-所有关系长度
        B = len(all_relations)
        _relation_bank = dict()
        _relation_length = dict()
        for k, v in all_relations.items():
            # _relation_bank-关系编号{relation_id: path relation id}
            _relation_bank[v] = np.array(k, dtype=np.int)
            # _relation_length-关系长度{relation_id: length}
            _relation_length[v] = len(k)
        # 根据关系编号id排序-'<CLS>'...
        # _relation_bank-顺序id [len(all_relations)), n]
        _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
        # _relation_length-长度 [len(all_relations))]
        _relation_length = [_relation_length[i] for i in range(len(all_relations))]
        #  _relation_bank-[len(all_relations)), n_max]顺序id
        _relation_bank = ArraysToTensor(_relation_bank).t_()
        _relation_length = torch.LongTensor(_relation_length)
    else:
        all_relations = dict()
        cls_idx = vocabs['relation'].token2idx(CLS)
        rcls_idx = vocabs['relation'].token2idx(rCLS)
        self_idx = vocabs['relation'].token2idx(SEL)
        pad_idx = vocabs['relation'].token2idx(PAD)
        all_relations[tuple([pad_idx])] = 0
        all_relations[tuple([cls_idx])] = 1
        all_relations[tuple([rcls_idx])] = 2
        all_relations[tuple([self_idx])] = 3

        _relation_type = []
        record = []
        bsz, num_concepts, num_paths = 0, 0, 0
        for bidx, x in enumerate(data):
            n = len(x['concept'])
            num_concepts = max(n+1, num_concepts)
            brs = [ [[3]]+[[1]]*(n) ]
            for i in range(n):
                rs = [[2]]
                for j in range(n):
                    all_r = []
                    all_path = x['relation'][str(i)][str(j)]
                    path0 = all_path[0]['edge']
                    if len(path0) == 0 or len(path0) > 8:
                        all_path = all_path[:1]
                    for path in all_path:
                        path = path['edge']
                        if len(path) == 0: # self loop
                            path = [SEL]
                        if len(path) > 8: # too long distance
                            path = [TL]
                        path = tuple(vocabs['relation'].token2idx(path))
                        rtype = all_relations.get(path, len(all_relations))
                        if rtype == len(all_relations):
                            all_relations[path] = len(all_relations)
                        all_r.append(rtype)
                    record.append(len(all_r))
                    num_paths = max(len(all_r), num_paths)
                    rs.append(all_r)
                brs.append(rs)
            _relation_type.append(brs)
        bsz = len(_relation_type) 
        _relation_matrix = np.zeros((bsz, num_concepts, num_concepts, num_paths))
        for b, x in enumerate(_relation_type):
            for i, y in enumerate(x):
                for j, z in enumerate(y):
                    for k, r in enumerate(z):
                        _relation_matrix[b, i, j, k] = r
        _relation_type = torch.from_numpy(_relation_matrix).transpose_(0, 2).long()

        B = len(all_relations)
        _relation_bank = dict()
        _relation_length = dict()
        for k, v in all_relations.items():
            _relation_bank[v] = np.array(k, dtype=np.int)
            _relation_length[v] = len(k)
        _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
        _relation_length = [_relation_length[i] for i in range(len(all_relations))]
        _relation_bank = ArraysToTensor(_relation_bank).t_()
        _relation_length = torch.LongTensor(_relation_length)

    local_token2idx = [x['token2idx'] for x in data]
    local_idx2token = [x['idx2token'] for x in data]

    # augmented_token-['<STR>', 'token', '<END>']-[batch_size, n]
    augmented_token = [[STR]+x['token']+[END] for x in data]

    # 将token转成id并转成张量
    # _token_in-[batch_size, n_max]
    _token_in = ListsToTensor(augmented_token, vocabs['token'], unk_rate=unk_rate)[:-1]
    # _token_char_in-[batch_size, n_max]
    _token_char_in = ListsofStringToTensor(augmented_token, vocabs['token_char'])[:-1]

    # _token_out-使用vocabs['predictable_token']和local_token2idx对token进行编号
    _token_out = ListsToTensor(augmented_token, vocabs['predictable_token'], local_token2idx)[1:]
    # _cp_seq-x['cp_seq']
    _cp_seq = ListsToTensor([ x['cp_seq'] for x in data], vocabs['predictable_token'], local_token2idx)

    # abstract-[batch_size, n_max]
    abstract = [ x['abstract'] for x in data]

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
    ret = {
        'concept': _conc,
        'concept_char': _conc_char,
        'concept_depth': _depth,
        'relation': _relation_type,
        'relation_bank': _relation_bank,
        'relation_length': _relation_length,
        'local_idx2token': local_idx2token,
        'local_token2idx': local_token2idx,
        'token_in':_token_in,
        'token_char_in':_token_char_in,
        'token_out':_token_out,
        'cp_seq': _cp_seq,
        'abstract': abstract
    }
    return ret

class DataLoader(object):
    def __init__(self, vocabs, lex_map, filename, batch_size, for_train):
        # 词典vocabs, 新词映射 LexicalMap, train_batch_size: 6666, for_train: True
        # ${dataset}/train.txt.features.preproc.json
        self.data = json.load(open(filename, encoding='utf8'))
        for d in self.data:
            # cp_seq: d['concept']词列表, 未在词典中的词新编号: token2idx, idx2token
            cp_seq, token2idx, idx2token = lex_map.get(d['concept'], vocabs['predictable_token'])
            d['cp_seq'] = cp_seq
            d['token2idx'] = token2idx
            d['idx2token'] = idx2token
        print ("Get %d AMR-English pairs from %s"%(len(self.data), filename))
        # 词典vocabs
        self.vocabs = vocabs
        # batch_size6666
        self.batch_size = batch_size
        # train: True
        self.train = for_train 
        self.unk_rate = 0.
        self.record_flag = False

    def set_unk_rate(self, x):
        # unk_rate: x
        self.unk_rate = x

    def record(self):
        # record_flag: True
        self.record_flag = True

    def __iter__(self):
        # [0, 1, ..., len-1]
        idx = list(range(len(self.data)))

        # 训练
        if self.train:
            # 打乱idx
            random.shuffle(idx)
            # 根据'token'长度
            idx.sort(key = lambda x: len(self.data[x]['token']) + len(self.data[x]['concept'])**2)

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            # token数目:['token']+['concept'])**2
            num_tokens += len(self.data[i]['token']) + len(self.data[i]['concept'])**2
            data.append(self.data[i])
            # batch_size:256
            if num_tokens >= self.batch_size or len(data)>256:
                # batches [data{}]
                # 一个data是一个batch
                batches.append(data)
                num_tokens, data = 0, []

        # 测试阶段或者num_tokens > self.batch_size/2
        if not self.train or num_tokens > self.batch_size/2:
            batches.append(data)

        # 打乱batch
        if self.train:
            random.shuffle(batches)

        for batch in batches:
            if not self.record_flag:
                # record_flag False
                # 构建生成器, 用于每次迭代输出
                # batch:[data{}], 词典vocabs, unk_rate 0, train True
                yield batchify(batch, self.vocabs, self.unk_rate, self.train)
            else:
                yield batchify(batch, self.vocabs, self.unk_rate, self.train), batch


def parse_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_vocab', type=str, default='../data/AMR/amr_2.0/token_vocab')
    parser.add_argument('--concept_vocab', type=str, default='../data/AMR/amr_2.0/concept_vocab')
    parser.add_argument('--predictable_token_vocab', type=str, default='../data/AMR/amr_2.0/predictable_token_vocab')
    parser.add_argument('--token_char_vocab', type=str, default='../data/AMR/amr_2.0/token_char_vocab')
    parser.add_argument('--concept_char_vocab', type=str, default='../data/AMR/amr_2.0/concept_char_vocab')
    parser.add_argument('--relation_vocab', type=str, default='../data/AMR/amr_2.0/relation_vocab')

    parser.add_argument('--train_data', type=str, default='../data/AMR/amr_2.0/dev.txt.features.preproc.json')
    parser.add_argument('--train_batch_size', type=int, default=10)

    return parser.parse_args()

if __name__ == '__main__':
    from extract import LexicalMap
    args = parse_config()
    vocabs = dict()
    # 构建词典集, 包含很多属性(size, padding)
    vocabs['concept'] = Vocab(args.concept_vocab, 5, [CLS])
    vocabs['token'] = Vocab(args.token_vocab, 5, [STR, END])
    vocabs['predictable_token'] = Vocab(args.predictable_token_vocab, 5, [END])
    vocabs['token_char'] = Vocab(args.token_char_vocab, 100, [STR, END])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 100, [STR, END])
    vocabs['relation'] = Vocab(args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
    # 新词映射 
    lexical_mapping = LexicalMap()

    # 词典vocabs, 新词映射 LexicalMap, train_batch_size: 6666, for_train: True
    train_data = DataLoader(vocabs, lexical_mapping, args.train_data, args.train_batch_size, for_train=True)
    epoch_idx = 0
    batch_idx = 0
    while True:
        for d in train_data:
            batch_idx += 1
            if d['concept'].size(0) > 5:
                continue
            print (epoch_idx, batch_idx, d['concept'].size(), d['token_in'].size())
            print (d['relation_bank'].size())
            print (d['relation'].size())

            _back_to_txt_for_check(d['concept'], vocabs['concept'])
            for x in d['concept_depth'].t().tolist():
                print (x)
            _back_to_txt_for_check(d['token_in'], vocabs['token'])
            _back_to_txt_for_check(d['token_out'], vocabs['predictable_token'], d['local_idx2token'])
            _back_to_txt_for_check(d['cp_seq'], vocabs['predictable_token'], d['local_idx2token'])
            _back_to_txt_for_check(d['relation_bank'], vocabs['relation'])
            print (d['relation'][:,:,0])
            exit(0)



