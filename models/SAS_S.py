import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from attention import*
from feedforward import*
from transformer import*
from base.seqential import*
from compress.STTD import STTD

class SAS_S(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, n_node, config):
        super(SAS_S, self).__init__(n_node, config)

        # load parameters info
        self.n_items = n_node
        self.n_layers = config.num_layer
        self.n_heads = config.num_heads
        self.hidden_size = config.hidden_units  # same as embedding_size
        self.emb_size = config.hidden_units
        self.inner_size = config.inner_units # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.dropout_rate
        self.attn_dropout_prob = config.dropout_rate
        self.hidden_act = config.act
        self.layer_norm_eps = 1e-12
        self.max_seq_length = 300
        self.batch_size = config.batch_size

        self.initializer_range = 0.01

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        self.w = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_1_hot = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        self.w_2_hot = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1_hot = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2_hot = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_1_cold = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        self.w_2_cold = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1_cold = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2_cold = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.mlp = nn.Linear(2 * self.emb_size, self.emb_size, bias=False)
        self.mlp1 = nn.Linear(2 * self.emb_size, self.emb_size, bias=False)
        self.mlp2 = nn.Linear(2 * self.emb_size, self.emb_size, bias=False)
        self.mlp3 = nn.Linear(2 * self.emb_size, self.emb_size, bias=False)
        self.beta = config.beta
        self.para = config.para
        self.alpha = config.alpha
        self.relu = nn.ReLU()
        self.STTD = STTD(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # """ Initialize the weights """
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     # Slightly different from the TF version which uses truncated_normal for initialization
        #     # cf https://github.com/pytorch/pytorch/pull/5617
        #     module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()
        for weight in self.parameters():
            weight.data.uniform_(-0.1, 0.1)

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def generate_sess_emb(self, seq_h, seq_len, mask):
        hs = torch.div(torch.sum(seq_h, 1), seq_len)
        len = seq_h.shape[1]
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        mask = mask.float().unsqueeze(-1)
        sess = beta * mask
        sess_emb = torch.sum(sess * seq_h, 1)
        return sess_emb

    def generate_sess_emb_hot(self, item_seq, seq_len, mask):
        get = lambda i: self.embedding[item_seq[i]]
        seq_h = torch.cuda.FloatTensor(list(item_seq.shape)[0], list(item_seq.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(item_seq.shape)[1], self.emb_size)
        for i in torch.arange(item_seq.shape[0]):
            seq_h[i] = get(i)
        hs = torch.sum(seq_h, 1) / seq_len
        len = seq_h.shape[1]
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh.float())
        nh = torch.sigmoid(self.glu1_hot(nh) + self.glu2_hot(hs))
        beta = torch.matmul(nh, self.w_2_hot)
        mask = mask.float().unsqueeze(-1)
        sess = beta * mask
        sess_emb = torch.sum(sess * seq_h, 1)
        return sess_emb

    def generate_sess_emb_cold(self, item_seq, seq_len, mask):
        get = lambda i: self.embedding[item_seq[i]]
        seq_h = torch.cuda.FloatTensor(list(item_seq.shape)[0], list(item_seq.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(item_seq.shape)[1], self.emb_size)
        for i in torch.arange(item_seq.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), seq_len)
        len = seq_h.shape[1]
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1_cold(nh) + self.glu2_cold(hs))
        beta = torch.matmul(nh, self.w_2_cold)
        mask = mask.float().unsqueeze(-1)
        sess = beta * mask
        sess_emb = torch.sum(sess * seq_h, 1)
        return sess_emb

    def predictive(self, hot_sess_stu, hot_sess_tea, cold_sess_stu, cold_sess_tea, hot_tar, cold_tar, cold_only_sess_stu, cold_only_sess_tea, hot_only_sess_stu, hot_only_sess_tea, hot_only_tar, teacher):

        sess_emb_stu = torch.cat((hot_sess_stu, cold_sess_tea), 1)
        sess_emb_tea = torch.cat((hot_sess_tea, cold_sess_stu), 1)
        sess_emb_stu = self.mlp2(sess_emb_stu)
        sess_emb_tea = self.mlp3(sess_emb_tea)

        sess_emb_stu = torch.cat([sess_emb_stu, cold_only_sess_tea, hot_only_sess_tea], 0)
        sess_emb_tea = torch.cat([sess_emb_tea, cold_only_sess_stu, hot_only_sess_stu], 0)
        sess_emb_stu = fn.normalize(sess_emb_stu, p=2, dim=-1)
        sess_emb_tea = fn.normalize(sess_emb_tea, p=2, dim=-1)

        tar = torch.cat([hot_tar, cold_tar, hot_only_tar], 0)

        loss = self.loss_fct(torch.mm(sess_emb_stu, torch.transpose(self.embedding, 1, 0)), tar)

        loss += self.loss_fct(torch.mm(sess_emb_tea, torch.transpose(teacher.embedding.weight, 1, 0)), tar)

        return loss

    def PredLoss(self, score_teacher, score_student):
        score_teacher = fn.softmax(score_teacher, dim=1)
        score_student = fn.softmax(score_student, dim=1)
        loss = torch.sum(torch.mul(score_teacher, torch.log(1e-8 + ((score_teacher + 1e-8)/(score_student + 1e-8)))))
        return loss

    def SSL(self, hot_sess_stu, hot_sess_tea, cold_sess_stu, cold_sess_tea, cold_only_stu, cold_only_tea, hot_only_stu, hot_only_tea):

        sess_emb_stu = torch.cat((hot_sess_stu, cold_sess_tea), 1)
        sess_emb_tea = torch.cat((hot_sess_tea, cold_sess_stu), 1)

        sess_emb_stu = self.mlp(sess_emb_stu)
        sess_emb_tea = self.mlp1(sess_emb_tea)

        sess_emb_stu = torch.cat([sess_emb_stu, cold_only_tea, hot_only_tea], 0)
        sess_emb_tea = torch.cat([sess_emb_tea, cold_only_stu, hot_only_stu], 0)

        sess_emb_tea = fn.normalize(sess_emb_tea,dim=-1,p=2)
        sess_emb_stu = fn.normalize(sess_emb_stu, dim=-1, p=2)

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_stu, sess_emb_tea)
        neg = torch.mm(sess_emb_stu, torch.transpose(sess_emb_tea, 1, 0))
        pos_score = torch.exp(pos / 0.2)
        neg_score = torch.sum(torch.exp(neg / 0.2), 1)
        # print('pos score:', pos_score, 'neg_score:', neg_score)
        con_loss = -torch.mean(torch.sum(torch.log((pos_score + 1e-8) / (neg_score + 1e-8) + 1e-8), -1))

        return con_loss

    def forward(self, item_seq, item_seq_len, mask):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        self.embedding = STTD()

        get = lambda i: self.embedding[item_seq[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(item_seq.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(item_seq.shape)[1], self.emb_size)
        for i in torch.arange(item_seq.shape[0]):
            seq_h[i] = get(i)
        item_emb = seq_h
        item_emb = item_emb.reshape(self.batch_size, -1, self.emb_size)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        self.output = self.generate_sess_emb(output, item_seq_len, mask)
        return self.output, self.embedding

    def forward_test(self, item_seq, item_seq_len, mask):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        get = lambda i: self.embedding[item_seq[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(item_seq.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(item_seq.shape)[1], self.emb_size)
        for i in torch.arange(item_seq.shape[0]):
            seq_h[i] = get(i)
        item_emb = seq_h
        item_emb = item_emb.reshape(self.batch_size, -1, self.emb_size)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.generate_sess_emb(output, item_seq_len, mask)
        return output, self.embedding

    def interact(self, item_seq, item_seq_len, mask, hot_sess_items, cold_sess_items, hot_sess_len, cold_sess_len, hot_cold_tar, hot_mask, cold_mask, hot_only_index, cold_only_index, teacher, tar):
        teacher.eval()
        _, _ = teacher(item_seq, item_seq_len, mask)

        teacher_score = torch.matmul(teacher.output, teacher.embedding.weight.transpose(1, 0))

        hot_sess_stu = self.generate_sess_emb_hot(hot_sess_items, hot_sess_len, hot_mask)
        cold_sess_stu = self.generate_sess_emb_cold(cold_sess_items, cold_sess_len, cold_mask)
        hot_sess_tea = teacher.generate_sess_emb_hot(hot_sess_items, hot_sess_len, hot_mask)
        cold_sess_tea = teacher.generate_sess_emb_cold(cold_sess_items, cold_sess_len, cold_mask)

        con_loss = self.SSL(hot_sess_stu, hot_sess_tea, cold_sess_stu, cold_sess_tea, self.output[cold_only_index],teacher.output[cold_only_index], self.output[hot_only_index], teacher.output[hot_only_index])
        #
        pre_loss = self.predictive(hot_sess_stu, hot_sess_tea, cold_sess_stu, cold_sess_tea,hot_cold_tar,tar[cold_only_index],self.output[cold_only_index], self.output[hot_only_index], teacher.output[hot_only_index],teacher.output[cold_only_index],tar[hot_only_index], teacher)

        loss_pre = self.PredLoss(teacher_score, torch.matmul(self.output, self.embedding.transpose(1, 0)))
        return self.para*loss_pre + self.beta*con_loss + self.alpha * pre_loss

    def full_sort_predict(self, item_seq, seq_len, mask):
        seq_output, item_emb = self.forward_test(item_seq, seq_len, mask)
        scores = torch.matmul(seq_output, item_emb.transpose(0, 1))
        return scores

