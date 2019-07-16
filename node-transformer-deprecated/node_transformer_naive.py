import torchdiffeq
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.Modules import ScaledDotProductAttention
from transformer.Models import (
    Decoder, get_attn_key_pad_mask, get_non_pad_mask, get_sinusoid_encoding_table, get_subsequent_mask
)
from transformer.SubLayers import PositionwiseFeedForward

from odeint_ext import odeint_adjoint_ext as odeint

class NodeMultiHeadAttentionFunc(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, t, qkv, mask):
        q, k, v = qkv
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        #return output, attn
        return output
    

    
class NodeMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, method='dopri5', rtol=1e-3, atol=1e-3):
        super().__init__()

        self.method = method
        self.node_func = NodeMultiHeadAttentionFunc(n_head, d_model, d_k, d_v)
        self.layer_norm = nn.LayerNorm(d_model)
        self.rtol = rtol
        self.atol = atol
    
    def forward(self, q, k, v, ts, mask):
        #q = q.unsqueeze(0)
        #k = k.unsqueeze(0)
        #v = v.unsqueeze(0)
        qkv = torch.stack((q, k, v), dim=0)
        #output, attn = odeint(self.node_func, qkv, ts, method=self.method, options={"mask":mask},
        #                      rtol=1e-3, atol=1e-3)
        output = odeint(self.node_func, qkv, ts, method=self.method, options={"mask":mask},
                              rtol=self.rtol, atol=self.atol)
        # output contains approximation of q, k, v so we keep only v which is the value we want to keep
        #output = output[2]
        # keep only last element (last timestamp)
        #output = output[-1, :, :]
        output = output[-1, 0, :]
        output = self.layer_norm(output)

        #return output, attn
        return output
    
    
class NodeEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, method='dopri5', rtol=1e-3, atol=1e-3):
        super(NodeEncoderLayer, self).__init__()
        self.slf_attn = NodeMultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, method=method, rtol=rtol, atol=atol)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, ts, non_pad_mask=None, slf_attn_mask=None):
        #enc_output, enc_slf_attn = self.slf_attn(
        #    enc_input, enc_input, enc_input, ts, mask=slf_attn_mask)
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input, ts, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        #return enc_output, enc_slf_attn
        return enc_output

        return output
    
class NodeEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, method='dopri5', rtol=1e-3, atol=1e-3):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            NodeEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, method=method, rtol=rtol, atol=atol)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, ts, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            #enc_output, enc_slf_attn = enc_layer(
            #    enc_output, ts,
            #    non_pad_mask=non_pad_mask,
            #    slf_attn_mask=slf_attn_mask)
            enc_output = enc_layer(
                enc_output, ts,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output

    

class NodeEncoderLayerFunc(nn.Module):
    def __init__(self, n_head, d_model, d_inner, d_k, d_v, dropout=0.1):
        super().__init__()

        self.mha_func = NodeMultiHeadAttentionFunc(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        
    def forward(self, t, enc_input, non_pad_mask, slf_attn_mask):
        qkv = torch.stack((enc_input, enc_input, enc_input), dim=0)
        output = self.mha_func(t, qkv, mask=slf_attn_mask)
        output *= non_pad_mask
        output = self.layer_norm(output)

        output = self.pos_ffn(output)
        output *= non_pad_mask
        output = self.layer_norm(output)
        return output    

    
class NodeEncoderLayer2(nn.Module):
    def __init__(self, n_head, d_model, d_inner, d_k, d_v, dropout=0.1, method='dopri5', rtol=1e-3, atol=1e-3):
        super().__init__()

        self.method = method
        self.node_func = NodeEncoderLayerFunc(n_head, d_model, d_inner, d_k, d_v, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.rtol = rtol
        self.atol = atol
        
        
    def forward(self, enc_input, ts, non_pad_mask=None, slf_attn_mask=None):
        output = odeint(self.node_func, enc_input, ts,
                        method=self.method, options={"non_pad_mask":non_pad_mask, "slf_attn_mask":slf_attn_mask},
                        rtol=self.rtol, atol=self.atol)
        # keep only last time step
        output = output[-1, :, :]
        output = self.layer_norm(output)
        return output


class NodeEncoder2(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, method='dopri5', rtol=1e-3, atol=1e-3):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.encoder = NodeEncoderLayer2(
            n_head, d_model, d_inner, d_k, d_v, dropout=dropout, method=method, rtol=rtol, atol=atol)

    def forward(self, src_seq, src_pos, ts, return_attns=False):

#        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        enc_output = self.encoder(
                enc_output, ts,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        
#        if return_attns:
#            return enc_output, enc_slf_attn_list
        return enc_output
    

class NodeDecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, method='dopri5', rtol=1e-3, atol=1e-3):
        super(NodeDecoderLayer, self).__init__()
        self.slf_attn = NodeMultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, method=method, rtol=rtol, atol=atol)
        self.enc_attn = NodeMultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, method=method, rtol=rtol, atol=atol)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, ts, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        #dec_output, dec_slf_attn = self.slf_attn(
        #    dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output = self.slf_attn(
            dec_input, dec_input, dec_input, ts, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        #dec_output, dec_enc_attn = self.enc_attn(
        #    dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.enc_attn(
            dec_output, enc_output, enc_output, ts, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        #return dec_output, dec_slf_attn, dec_enc_attn
        return dec_output


    
    
    
class NodeDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, method='dopri5', rtol=1e-3, atol=1e-3):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            NodeDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, method=method, rtol=rtol, atol=atol)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, ts, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            #dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
            dec_output = dec_layer(
                dec_output, enc_output, ts,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        #return dec_output,
        return dec_output
    
    
    
class NodeTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True,
            method='dopri5', rtol=1e-3, atol=1e-3):

        super().__init__()

        self.encoder = NodeEncoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, method=method, rtol=rtol, atol=atol)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        #self.decoder = NodeDecoder(
        #    n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
        #    d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
        #    n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
        #    dropout=dropout, method=method, rtol=rtol, atol=atol)        
        
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, ts):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        #enc_output, *_ = self.encoder(src_seq, src_pos, ts)
        enc_output = self.encoder(src_seq, src_pos, ts)
        #print("enc_ouput", enc_output.size())
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        #dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, ts)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
    