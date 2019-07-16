import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import torchdiffeq

from tensorboard_utils import Tensorboard
from tensorboard_utils import tensorboard_event_accumulator

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.Modules import ScaledDotProductAttention
from transformer.Models import Decoder, get_attn_key_pad_mask, get_non_pad_mask, get_sinusoid_encoding_table
from transformer.SubLayers import PositionwiseFeedForward

import dataset

import model_process
import checkpoints
from node_transformer import NodeTransformer
from odeint_ext import SOLVERS

from itertools import islice

print("Torch Version", torch.__version__)
print("Solvers", SOLVERS)

seed = 1
torch.manual_seed(seed)
device = torch.device("cuda")
print("device", device)


data = torch.load("/home/mandubian/datasets/multi30k/multi30k.atok.low.pt")

max_token_seq_len = data['settings'].max_token_seq_len
print("max_token_seq_len", max_token_seq_len)

train_loader, val_loader = dataset.prepare_dataloaders(data, batch_size=128, num_workers=0)

src_vocab_sz = train_loader.dataset.src_vocab_size
print("src_vocab_sz", src_vocab_sz)
tgt_vocab_sz = train_loader.dataset.tgt_vocab_size
print("tgt_vocab_sz", tgt_vocab_sz)

exp_name = "node_transformer_dopri5_multi30k"
unique_id = "2019-06-10_1100"

model = NodeTransformer(
    n_src_vocab=max(src_vocab_sz, tgt_vocab_sz),
    n_tgt_vocab=max(src_vocab_sz, tgt_vocab_sz),
    len_max_seq=max_token_seq_len,
    #emb_src_tgt_weight_sharing=False,
    #d_word_vec=64, d_model=64, d_inner=256,
    n_head=8, method='dopri5-ext', rtol=1e-2, atol=1e-2,
    has_node_encoder=True, has_node_decoder=True)

model = model.to(device)

#tb = Tensorboard(exp_name, unique_name=unique_id)

optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.995), eps=1e-9)

# Continuous space discretization
timesteps = np.linspace(0., 1, num=6)
timesteps = torch.from_numpy(timesteps).float()

EPOCHS = 1
LOG_INTERVAL = 5

train_loader = list(islice(train_loader, 0, 20))

model_process.train(
    exp_name, unique_id,
    model, 
    train_loader, val_loader, timesteps,
    optimizer, device,
    epochs=EPOCHS, tb=None, log_interval=LOG_INTERVAL,
    start_epoch=0 #, best_valid_accu=state["acc"]
)
