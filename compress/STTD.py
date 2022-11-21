import math
import torch
import torch.nn as nn
import torch.nn.functional as fn
from utils.cuda import*

class STTD(nn.Module):

    def __init__(self, config):
        super(STTD, self).__init__(config)
        self.t = config.t
        self.STT = config.STT
        self.block = config.blocks  # (12, 38, 88)(8, 4, 4)
        self.block_num = config.b_num
        self.tt_rank = {}
        for i in range(self.block_num):
            if i == 0:
                self.tt_rank[i] = 1
            else:
                self.tt_rank[i] = config.tt_rank
        self.tt_rank[self.block_num] = 1

        if self.block_num == 2 and config.STT is True:
            dim1 = self.block[0][0]
            dim2 = self.block[1][0]
            # self.emb0 = nn.Parameter(torch.Tensor(self.tt_rank[0]*dim1*dim2, self.tt_rank[1]))
            self.emb0 = nn.Embedding(self.tt_rank[0] * dim1 * dim2, self.tt_rank[1])
            dim1 = self.block[0][1]
            dim2 = self.block[1][1]
            # self.emb1 = nn.Parameter(torch.Tensor(int(self.tt_rank[1]/config.t), dim1*dim2*int(self.tt_rank[2]/config.t)))
            self.emb1 = nn.Embedding(int(self.tt_rank[1] / config.t), int(dim1 * dim2 * self.tt_rank[2] / config.t))
        if self.block_num == 2 and config.STT is False:
            dim1 = self.block[0][0]
            dim2 = self.block[1][0]
            # self.emb0 = nn.Parameter(torch.Tensor(self.tt_rank[0], dim1, dim2, self.tt_rank[1]))
            self.emb0 = nn.Embedding(self.tt_rank[0] * dim1 * dim2, self.tt_rank[1])
            # print(self.emb0)
            dim1 = self.block[0][1]
            dim2 = self.block[1][1]
            # self.emb1 = nn.Parameter(torch.Tensor(self.tt_rank[1], dim1, dim2, self.tt_rank[2]))
            self.emb1 = nn.Embedding(self.tt_rank[1], dim1 * dim2 * self.tt_rank[2])
        if self.block_num == 3 and config.STT is True:
            dim1 = self.block[0][0]
            dim2 = self.block[1][0]
            self.emb0 = nn.Embedding(self.tt_rank[0] * dim1 * dim2, self.tt_rank[1])
            dim1 = self.block[0][1]
            dim2 = self.block[1][1]
            self.emb1 = nn.Embedding(int(self.tt_rank[1] / config.t), int(dim1 * dim2 * self.tt_rank[2] / config.t))
            dim1 = self.block[0][2]
            dim2 = self.block[1][2]
            self.emb2 = nn.Embedding(int(self.tt_rank[2] / config.t), int(dim1 * dim2 * self.tt_rank[3] / config.t))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        for weight in self.parameters():
            weight.data.uniform_(-0.1, 0.1)

    def get_emb(self):
        tt_rank = self.tt_rank[1]
        if self.block_num == 1:
            emb = self.emb0
        elif self.block_num == 2:
            emb = torch.mm(self.emb0.weight.reshape(-1, tt_rank), self.emb1.weight.reshape(tt_rank, -1))
        elif self.block_num == 3:
            emb = torch.mm(self.emb0.weight.reshape(-1, tt_rank), self.emb1.weight.reshape(tt_rank, -1))
            emb = torch.mm(emb.reshape(-1, tt_rank), self.emb2.weight.reshape(tt_rank, -1))
        else:
            emb = torch.mm(self.emb0.reshape(-1, tt_rank), self.emb1.reshape(tt_rank, -1))
            emb = torch.mm(emb.reshape(-1, tt_rank), self.emb2.reshape(tt_rank, -1))
            emb = torch.mm(emb.reshape(-1, tt_rank), self.emb3.reshape(tt_rank, -1))
        return emb.reshape(self.n_items, self.emb_size)

    def merge_STT(self,emb0, emb1, t):
        N = emb0.shape[1]
        slice = trans_to_cuda(torch.stack(torch.arange(0, N).split(t, dim=-1)))
        emb = {}
        for i in range(t):
            emb[i] = torch.mm(torch.index_select(emb0, dim=-1, index=slice[:, i]), emb1)
        embedding = emb[0]
        for i in range(1, t):
            embedding = torch.cat([embedding, emb[i]], dim=-1)

        embedding = embedding.transpose(0, 1)
        slice = []
        for i in range(int(embedding.shape[0]/2)):
            slice += [i, i + int(embedding.shape[0]/2)]
        slice = trans_to_cuda(torch.tensor(slice).long())
        embedding = embedding[slice]
        del slice, emb
        return embedding.transpose(0, 1).reshape(-1, self.emb_size)

    def merge_STT_3(self, emb0, emb1, emb2, t):
        N = emb0.shape[1]
        slice = trans_to_cuda(torch.stack(torch.arange(0, N).split(t, dim=-1)))
        emb = {}
        for i in range(t):
            emb[i] = torch.mm(torch.index_select(emb0, dim=-1, index=slice[:, i]), emb1)
        embedding = emb[0]
        for i in range(1, t):
            embedding = torch.cat([embedding, emb[i]], dim=-1)

        embedding = embedding.transpose(0, 1)
        slice = []
        for i in range(int(embedding.shape[0] / 2)):
            slice += [i, i + int(embedding.shape[0] / 2)]
        slice = trans_to_cuda(torch.tensor(slice).long())
        embedding = embedding[slice]
        del slice, emb
        embedding = embedding.transpose(0, 1).reshape(-1, self.tt_rank[2])

        emb0 = embedding
        emb1 = emb2
        N = emb0.shape[1]
        slice = trans_to_cuda(torch.stack(torch.arange(0, N).split(t, dim=-1)))
        emb = {}
        for i in range(t):
            emb[i] = torch.mm(torch.index_select(emb0, dim=-1, index=slice[:, i]), emb1)
        embedding = emb[0]
        for i in range(1, t):
            embedding = torch.cat([embedding, emb[i]], dim=-1)

        embedding = embedding.transpose(0, 1)
        slice = []
        for i in range(int(embedding.shape[0] / 2)):
            slice += [i, i + int(embedding.shape[0] / 2)]
        slice = trans_to_cuda(torch.tensor(slice).long())
        embedding = embedding[slice]
        del slice, emb
        embedding = embedding.transpose(0, 1).reshape(-1, self.emb_size)
        return embedding

    def forward(self):
        if self.block_num == 2 and self.STT == True:
            embedding = self.merge_STT(self.emb0.weight, self.emb1.weight, self.t)
        elif self.block_num == 3 and self.STT == True:
            embedding = self.merge_STT_3(self.emb0.weight, self.emb1.weight, self.emb2.weight, self.t)
        elif not self.STT:
            embedding = self.get_emb()
        return embedding