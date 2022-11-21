from transformer import*
from base.seqential import*


class SAS_T(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, n_node, config):
        super(SAS_T, self).__init__(n_node, config)

        # load parameters info
        self.n_items = n_node
        self.n_layers = config.num_layer
        self.n_heads = config.num_heads
        self.emb_size = config.hidden_units
        self.hidden_size = config.hidden_units  # same as embedding_size
        self.inner_size = config.inner_units # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.dropout_rate
        self.attn_dropout_prob = config.dropout_rate
        self.hidden_act = config.act
        self.layer_norm_eps = 1e-12
        self.max_seq_length = 300
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        # self.w_1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_1_hot = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        # self.w_1_hot = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.w_2_hot = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1_hot = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2_hot = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_1_cold = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        # self.w_1_cold = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.w_2_cold = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1_cold = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2_cold = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.initializer_range = 0.01

        # define layers and loss
        self.embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     # Slightly different from the TF version which uses truncated_normal for initialization
        #     # cf https://github.com/pytorch/pytorch/pull/5617
        #     module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            # weight.data.normal_(0, 0.1)
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
        seq_h = self.embedding(item_seq)
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
        seq_h = self.embedding(item_seq)
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

    def forward(self, item_seq, item_seq_len, mask):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        self.output = self.generate_sess_emb(output, item_seq_len, mask)
        # print(self.output)
        return self.output, self.embedding.weight # [B H]

    def full_sort_predict(self, item_seq, seq_len, mask):
        seq_output, item_emb = self.forward(item_seq, seq_len, mask)
        scores = torch.matmul(seq_output, item_emb.transpose(0, 1))  # [B n_items]
        return scores





