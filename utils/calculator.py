import sys
import copy
import random
import numpy as np
from collections import defaultdict
from operator import itemgetter


def count_hot(sessions, target):
    cnt_hot = {}
    for sess in sessions:
        for i in sess:
            if i not in cnt_hot:
                cnt_hot[i] = 1
            else:
                cnt_hot[i] += 1
    cnt_hot = sorted(cnt_hot.items(), key=lambda kv: kv[1], reverse=True)
    # hot_item = list(cnt_hot.keys())
    length = len(cnt_hot)
    hot = [cnt_hot[i][0] for i in range(int(length*0.2))]
    most_pop = [cnt_hot[i][0] for i in range(int(length*0.05))]
    return cnt_hot, hot, most_pop


def compress_ratio_TTD(opt, n_node):
    model_size = n_node * opt.hidden_units
    if opt.b_num == 2:
        compressed = opt.blocks[0][0] * opt.blocks[1][0] * opt.tt_rank + int(
            (opt.blocks[0][1] * opt.blocks[1][1] * opt.tt_rank) / (opt.t * opt.t))
        print('compression rate:', float(model_size) / compressed)
    if opt.b_num == 3:
        compressed = opt.blocks[0][0] * opt.blocks[1][0] * opt.tt_rank + int(
            (opt.blocks[0][1] * opt.blocks[1][1] * opt.tt_rank * opt.tt_rank) / (opt.t * opt.t)) + int(
            (opt.blocks[0][2] * opt.blocks[1][2] * opt.tt_rank) / (opt.t * opt.t))
        print('compression rate:', float(model_size) / compressed)


def compress_ratio_codebook(opt, n_node):
    print('compression rate:', (n_node * opt.hidden_dim) / (n_node * opt.code_book_len + opt.code_book_len * opt.cluster_num * opt.hidden_dim))