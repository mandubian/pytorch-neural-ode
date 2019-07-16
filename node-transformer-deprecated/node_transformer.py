import torchdiffeq
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.Modules import ScaledDotProductAttention
from transformer.Models import (
    Decoder, Encoder, get_attn_key_pad_mask, get_non_pad_mask, get_sinusoid_encoding_table, get_subsequent_mask
)
from transformer.SubLayers import PositionwiseFeedForward

from odeint_ext import odeint_adjoint_ext as odeint
#from odeint_ext import odeint_ext as odeint

class NodeMultiHeadAttentionFunc(nn.Module):
    ''' multi-head attention '''
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

    def forward(self, t, q, k, v, mask):
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
        #output = self.dropout(self.fc(output))
        output = self.fc(output)
        #return output, attn
        return output
    

class NodePositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        #output = self.dropout(output)
        return output
    

    
class NodeSelfAttentionLayerFunc(nn.Module):
    ''' multi-head attention + layer-norm function only '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, add_time=True):
        super(NodeSelfAttentionLayerFunc, self).__init__()
        self.slf_attn = NodeMultiHeadAttentionFunc(n_head, d_model, d_k, d_v, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.nfe = 0
        self.add_time = add_time

    def forward(self, t, input, non_pad_mask=None, slf_attn_mask=None):
        self.nfe += 1

        # add time to input
        if self.add_time:
            input = input + t
            
        output = self.slf_attn(
            t, input, input, input, mask=slf_attn_mask)
        output = self.layer_norm(output)
        output *= non_pad_mask

        return output
    

class NodeEncoderLayerFunc(nn.Module):
    ''' node-encoder : multi-head attention + position-wise-feed-forward + layer-norm function '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, add_time=True):
        super().__init__()

        self.mha_func = NodeMultiHeadAttentionFunc(n_head, d_model, d_k, d_v)
        self.pos_ffn = NodePositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.nfe = 0
        self.add_time = add_time

        
    def forward(self, t, enc_input, non_pad_mask, slf_attn_mask):
        #qkv = torch.stack((enc_input, enc_input, enc_input), dim=0)
        self.nfe += 1
        
        # add time to input
        if self.add_time:
            enc_input = enc_input + t
        
        output = self.mha_func(t, enc_input, enc_input, enc_input, mask=slf_attn_mask)
        output = self.layer_norm(output)
        output *= non_pad_mask

        output = self.pos_ffn(output)
        output = self.layer_norm(output)
        output *= non_pad_mask
        return output    

    
class NodeEncoderLayer(nn.Module):
    ''' node-encoder layer that calls ODE-Solver on NodeEncoderLayerFunc '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, method='dopri5',
                 rtol=1e-3, atol=1e-3, add_time=True):
        super().__init__()

        self.node_func = NodeEncoderLayerFunc(d_model, d_inner, n_head, d_k, d_v, dropout, add_time=add_time)
        #self.layer_norm = nn.LayerNorm(d_model)
        self.method = method
        self.rtol = rtol
        self.atol = atol
        
        
    def forward(self, enc_input, ts, non_pad_mask=None, slf_attn_mask=None):
        output = odeint(self.node_func, enc_input, ts,
                        method=self.method,
                        options={"non_pad_mask":non_pad_mask, "slf_attn_mask":slf_attn_mask},
                        rtol=self.rtol, atol=self.atol)
        # keep only last time step
        output = output[-1, :, :]
        #output = self.layer_norm(output)
        return output

    @property
    def nfe(self):
        return self.node_func.nfe
    
    def reset_nfe(self):
        self.node_func.nfe = 0
    
class NodeEncoder(nn.Module):
    ''' A encoder model with NODE self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            d_model, d_inner,
            n_head, d_k, d_v,
            dropout=0.1, method='dopri5', rtol=1e-3, atol=1e-3, add_time=True):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.encoder = NodeEncoderLayer(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout, method=method, rtol=rtol, atol=atol, add_time=add_time)

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
    
    @property
    def nfe(self):
        return self.encoder.nfe    

    def reset_nfe(self):
        self.encoder.reset_nfe()
    
class NodeDecoderLayerFunc(nn.Module):
    ''' node-encoder : multi-head self-attention + multi-head encoder/decoder attention + position-wise-feed-forward + layer-norm function '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, add_time=True):
        super(NodeDecoderLayerFunc, self).__init__()
        self.slf_attn = NodeMultiHeadAttentionFunc(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = NodeMultiHeadAttentionFunc(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = NodePositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.nfe = 0
        self.add_time = add_time

    def forward(self, t, dec_input, enc_output=None, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        self.nfe += 1

        # add time to input
        if self.add_time:
            dec_input = dec_input + t
            # we don't add to enc_output as it should have been added earlier
            # enc_output = enc_output + t
        
        dec_output = self.slf_attn(
            t, dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output = self.layer_norm(dec_output)
        dec_output *= non_pad_mask

        dec_output = self.enc_attn(
            t, dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.layer_norm(dec_output)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output = self.layer_norm(dec_output)
        dec_output *= non_pad_mask

        #return dec_output, dec_slf_attn, dec_enc_attn
        return dec_output


class NodeDecoderLayer(nn.Module):
    ''' node-decoder layer that calls ODE-Solver on NodeDecoderLayerFunc '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, method='dopri5',
                 rtol=1e-3, atol=1e-3, add_time=True):
        super().__init__()

        self.method = method
        self.node_func = NodeDecoderLayerFunc(d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, add_time=add_time)
        self.rtol = rtol
        self.atol = atol
        self.layer_norm = nn.LayerNorm(d_model)
        
        
    def forward(self, dec_input, enc_output, ts,
                non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        output = odeint(self.node_func, dec_input, ts,
                        method=self.method,                        
                        rtol=self.rtol, atol=self.atol,
                        options={
                            "enc_output":enc_output, "non_pad_mask":non_pad_mask, 
                            "slf_attn_mask":slf_attn_mask, "dec_enc_attn_mask":dec_enc_attn_mask
                        })
        # keep only last time step
        output = output[-1, :, :]
        return output
    
    @property
    def nfe(self):
        return self.node_func.nfe
    
    def reset_nfe(self):
        self.node_func.nfe = 0
    
    
class NodeDecoder(nn.Module):
    ''' A decoder model with NODE self attention mechanism. '''

    def __init__(
        self,
        n_tgt_vocab, len_max_seq, d_word_vec,
        d_model, d_inner,
        n_head, d_k, d_v,
        dropout=0.1, method='dopri5',
        rtol=1e-3, atol=1e-3, add_time=True):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.decoder = NodeDecoderLayer(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout, method=method,
            rtol=rtol, atol=atol, add_time=add_time)

        #self.layer_norm = nn.LayerNorm(d_model)

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

        dec_output = self.decoder(
            dec_output, enc_output, ts,
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask,
            dec_enc_attn_mask=dec_enc_attn_mask)

        #dec_output = self.layer_norm(dec_output)
        #if return_attns:
        #    dec_slf_attn_list += [dec_slf_attn]
        #    dec_enc_attn_list += [dec_enc_attn]


        #if return_attns:
        #    return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output
    
    @property
    def nfe(self):
        return self.decoder.nfe
    
    def reset_nfe(self):
        self.decoder.reset_nfe()

        

    
class NodeDecoderEncoderLayerFunc(nn.Module):
    ''' node-decoder-encode layer: attention mechanism between encoder output and decoder self-attention output + position-wise-feed-forward + layer-norm '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.enc_attn = NodeMultiHeadAttentionFunc(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = NodePositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.nfe = 0

    def forward(self, t, dec_output, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        self.nfe += 1
        
        dec_output = self.enc_attn(
            t, dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.layer_norm(dec_output)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output = self.layer_norm(dec_output)
        dec_output *= non_pad_mask

        return dec_output

class NodeSeparatedDecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, method='dopri5',
                 rtol=1e-3, atol=1e-3, add_time=True):
        super().__init__()

        self.method = method
        self.slf_attn = NodeSelfAttentionLayerFunc(d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, add_time=add_time)
        self.enc_attn = NodeDecoderEncoderLayerFunc(d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        self.rtol = rtol
        self.atol = atol
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, dec_input, enc_output, ts,
                non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output = odeint(self.slf_attn, dec_input, ts,
                        method=self.method,                        
                        rtol=self.rtol, atol=self.atol,
                        options={
                            "non_pad_mask":non_pad_mask, 
                            "slf_attn_mask":slf_attn_mask
                        })
        dec_output = self.layer_norm(dec_output)
        # keep only last time step
        dec_output = dec_output[-1, :, :]

        output = odeint(self.enc_attn, dec_output, ts,
                        method=self.method,                        
                        rtol=self.rtol, atol=self.atol,
                        options={
                            "enc_output": enc_output, "non_pad_mask":non_pad_mask, 
                            "slf_attn_mask":slf_attn_mask, "dec_enc_attn_mask":dec_enc_attn_mask
                        })        
        output = self.layer_norm(output)
        # keep only last time step
        output = output[-1, :, :]
        return output    

    @property
    def nfe(self):
        return self.slf_attn.nfe + self.enc_attn.nfe
    
    def reset_nfe(self):
        self.slf_attn.nfe = 0
        self.enc_attn.nfe = 0
            

class NodeSeparatedDecoder(nn.Module):
    ''' A decoder model with NODE self attention mechanism. '''

    def __init__(
        self,
        n_tgt_vocab, len_max_seq, d_word_vec,
        d_model, d_inner,
        n_head, d_k, d_v,
        dropout=0.1, method='dopri5',
        rtol=1e-3, atol=1e-3, add_time=True):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.decoder = NodeSeparatedDecoderLayer(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout, method=method,
            rtol=rtol, atol=atol, add_time=add_time)

        #self.layer_norm = nn.LayerNorm(d_model)

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

        dec_output = self.decoder(
            dec_output, enc_output, ts,
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask,
            dec_enc_attn_mask=dec_enc_attn_mask)

        #dec_output = self.layer_norm(dec_output)
        #if return_attns:
        #    dec_slf_attn_list += [dec_slf_attn]
        #    dec_enc_attn_list += [dec_enc_attn]


        #if return_attns:
        #    return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output
    
    @property
    def nfe(self):
        return self.decoder.nfe
    
    def reset_nfe(self):
        self.decoder.reset_nfe()
        
        
        
class NodeTransformer(nn.Module):
    ''' A sequence to sequence model with optional NODE attention mechanism. '''

    def __init__(
        self,
        n_src_vocab, n_tgt_vocab, len_max_seq,
        d_word_vec=512, d_model=512, d_inner=2048,
        n_layers=1, n_head=8, d_k=64, d_v=64, dropout=0.1,
        tgt_emb_prj_weight_sharing=True,
        emb_src_tgt_weight_sharing=True,
        method='dopri5', rtol=1e-3, atol=1e-3,
        has_node_encoder=True,
        has_node_decoder=True,
        has_separated_node_decoder=False,
        add_time=True,
    ):

        super().__init__()
        self.has_node_encoder = has_node_encoder
        self.has_node_decoder = has_node_decoder
        if self.has_node_encoder:
            self.encoder = NodeEncoder(
                n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_head=n_head, d_k=d_k, d_v=d_v,
                dropout=dropout, method=method, rtol=rtol, atol=atol, add_time=add_time)
        else:
            self.encoder = Encoder(
                n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                dropout=dropout)

        if self.has_node_decoder:
            if has_separated_node_decoder:
                self.decoder = NodeSeparatedDecoder(
                    n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
                    d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                    n_head=n_head, d_k=d_k, d_v=d_v,
                    dropout=dropout, method=method, rtol=rtol, atol=atol, add_time=add_time)
            else:
                self.decoder = NodeDecoder(
                    n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
                    d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                    n_head=n_head, d_k=d_k, d_v=d_v,
                    dropout=dropout, method=method, rtol=rtol, atol=atol, add_time=add_time)

        else:
            self.decoder = Decoder(
                n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                dropout=dropout)


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

        if self.has_node_encoder:
            enc_output = self.encoder(src_seq, src_pos, ts)
        else:
            enc_output, *_ = self.encoder(src_seq, src_pos)
            
        if self.has_node_decoder:
            dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, ts)
        else:
            dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))

    @property
    def nfes(self):
        if self.has_node_encoder:
            nfe_encoder = self.encoder.nfe
        else:
            nfe_encoder = 0

        if self.has_node_decoder:
            nfe_decoder = self.decoder.nfe
        else:
            nfe_decoder = 0

        return (nfe_encoder, nfe_decoder)

    def reset_nfes(self):
        if self.has_node_encoder:
            self.encoder.reset_nfe()
            
        if self.has_node_decoder:
            self.decoder.reset_nfe()
            
            
            

