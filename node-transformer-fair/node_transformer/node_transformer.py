# Copyright (c) 2019-present, Pascal Voitot
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

from fairseq.models.transformer  import (
    TransformerDecoder, TransformerEncoder
)

from fairseq.options import eval_str_list

from odeint_ext.odeint_ext import odeint_adjoint_ext as odeint


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MAX_NUM_STEPS=1000    
DEFAULT_AUGMENT_DIMS=0

@register_model('node_transformer')
class NodeTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        return {}

    def __init__(self, has_node_encoder, encoder, has_node_decoder, decoder):
        self.has_node_encoder = has_node_encoder
        self.has_node_decoder = has_node_decoder
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # Node Params
        parser.add_argument('--node-encoder', action='store_true',
                           help='Activate NODE Encoder')
        parser.add_argument('--node-decoder', action='store_true',
                           help='Activate NODE Decoder')
        parser.add_argument('--node-rtol', type=float, metavar='D', default=0.01,
                            help='ODE RTOL')
        parser.add_argument('--node-atol', type=float, metavar='D', default=0.01,
                            help='ODE ATOL')
        parser.add_argument('--node-method', type=str, metavar='STR', default="dopri5-ext",
                            help='ODE Method')
        parser.add_argument('--node-ts', '--node-time-steps', default='[0.0, 1.0]', type=eval_str_list,
                            metavar='TS_0,TS_1,...,TS_N',
                            help='Time Steps used by ODE SOlver')
        parser.add_argument('--node-max-num-steps', type=int, metavar='N', default=DEFAULT_MAX_NUM_STEPS,
                            help='ODE max number of steps in a solver operation')
        parser.add_argument('--node-augment-dims', type=int, metavar='N', default=DEFAULT_AUGMENT_DIMS,
                            help='ODE augmented dimensions')
        parser.add_argument('--node-time-dependent', action='store_true',
                           help='Activate node time dependency')
        parser.add_argument('--node-separated-decoder', action='store_true',
                           help='Activate node separation decoder')

        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        model = NodeTransformerModel(args.node_encoder, encoder, args.node_decoder, decoder)
        return model

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if args.node_encoder:
            return NodeTransformerEncoder(args, src_dict, embed_tokens)
        else:
            return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if args.node_decoder:
            return NodeTransformerDecoder(args, tgt_dict, embed_tokens)
        else:
            return TransformerDecoder(args, tgt_dict, embed_tokens)

    #@property
    #def nfe(self):
    #    nfe_enc = 0
    #    nfe_dec = 0
    #    if args.node_encoder:
    #        nfe_enc = self.encoder.nfe
    #    if args.node_decoder:
    #        nfe_dec = self.decoder.nfe
    #    return nfe_enc, nfe_dec
    
    #def reset_nfe(self):
    #    self.encoder.reset_nfe()
    #    self.decoder.reset_nfe()
        
class NodeTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None
        
        self.layers = nn.ModuleList([])
        self.layers.extend([
            NodeTransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        #x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def nfe(self, init_nfe=0):
        for i in range(len(self.layers)):
            init_nfe += self.layers[i].nfe()
        return init_nfe
    
    def reset_nfe(self):
        for i in range(len(self.layers)):
            self.layers[i].reset_nfe()
        
class NodeTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.augment_dims = args.node_augment_dims
        self.time_dependent = args.node_time_dependent
        
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        if not args.node_separated_decoder:
            self.layers.extend([
                NodeTransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ])
        else:
            self.layers.extend([
                NodeTransformerDecoderLayer_Separated(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ])

        self.adaptive_softmax = None

        # AUGMENTED NODE
        self.augment_embed_dim = embed_dim + self.augment_dims * args.decoder_attention_heads
        #if self.time_dependent:
        #    self.augment_embed_dim += 1 * args.decoder_attention_heads
        
        self.project_out_dim = Linear(self.augment_embed_dim, self.output_embed_dim, bias=False) \
            if self.augment_embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(self.augment_embed_dim)
        else:
            self.layer_norm = None

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
        x = self.output_layer(x)
        return x, extra

    def nfe(self, init_nfe=0):
        for i in range(len(self.layers)):
            init_nfe += self.layers[i].nfe()
        return init_nfe
    
    def reset_nfe(self):
        for i in range(len(self.layers)):
            self.layers[i].reset_nfe()    
    
    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        #x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

    
class NodeTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.method = args.node_method
        self.rtol = args.node_rtol
        self.atol = args.node_atol        
        self.func = NodeTransformerEncoderLayerFunc(args)
        self.ts = torch.FloatTensor(args.node_ts)
        self.max_num_steps = args.node_max_num_steps

    def forward(self, x, encoder_padding_mask):
        output = odeint(self.func, x, self.ts,
                        method=self.method,
                        options={"encoder_padding_mask":encoder_padding_mask,
                                 #'max_num_steps': self.max_num_steps,
                                },
                        rtol=self.rtol, atol=self.atol)
        # keep only last time step
        output = output[-1, :, :]
        return output

    def upgrade_state_dict_named(self, state_dict, name):
        # TODO Check if it should do more...
        self.func.upgrade_state_dict_named(state_dict, name)        
        return state_dict

    def nfe(self):
        return self.func.nfe
    
    def reset_nfe(self):
        self.func.nfe = 0    
    
class NodeTransformerEncoderLayerFunc(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout, self_attention=True
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.nfe = 0

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """        
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, t, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        self.nfe += 1
        #residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        #residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

                

class NodeTransformerDecoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.method = args.node_method
        self.rtol = args.node_rtol
        self.atol = args.node_atol        
        self.func = NodeTransformerDecoderLayerFunc(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.ts = torch.FloatTensor(args.node_ts)
        self.max_num_steps = args.node_max_num_steps
        self.augment_dims = args.node_augment_dims
        self.time_dependent = args.node_time_dependent
        self.decoder_attention_heads = args.decoder_attention_heads
        

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        if self.augment_dims > 0:
            seq_len, batch, embed_dim = x.shape
            aug = torch.zeros(seq_len, batch, self.augment_dims * self.decoder_attention_heads).to(x)
            # Shape (seq_len, batch, embed_dim + augment_dims)
            x = torch.cat([x, aug], 2)
            
            seq_len, batch, embed_dim = encoder_out.shape
            aug = torch.zeros(seq_len, batch, self.augment_dims * self.decoder_attention_heads).to(encoder_out)
            # Shape (seq_len, batch, embed_dim + augment_dims)
            encoder_out = torch.cat([encoder_out, aug], 2)
            
        output = odeint(self.func, x, self.ts,
                        method=self.method,
                        options={
                            "encoder_out": encoder_out,
                            "encoder_padding_mask": encoder_padding_mask,
                            "incremental_state": incremental_state,
                            "prev_self_attn_state": prev_self_attn_state,
                            "prev_attn_state": prev_attn_state,
                            "self_attn_mask": self_attn_mask,
                            "self_attn_padding_mask": self_attn_padding_mask,
                            #'max_num_steps': self.max_num_steps,
                        },
                        rtol=self.rtol, atol=self.atol)
        # keep only last time step
        output = output[-1, :, :]
        # customizing a bit because with ODE we can't return many values for ODE Solver
        if self.func.onnx_trace and incremental_state is not None:
            return output, self.func.cur_attn, self.func.cur_self_attn_state
        else:
            return output, self.func.cur_attn

    def nfe(self):
        return self.func.nfe
    
    def reset_nfe(self):
        self.func.nfe = 0    
            

class NodeTransformerDecoderLayerFunc(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.augment_dims = args.node_augment_dims
        self.time_dependent = args.node_time_dependent
        self.decoder_attention_heads = args.decoder_attention_heads
        
        self.embed_dim = args.decoder_embed_dim + self.augment_dims * args.decoder_attention_heads
        self.initial_embed_dim = self.embed_dim
        if self.time_dependent:
            self.embed_dim += 1 * args.decoder_attention_heads
            
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            kdim = getattr(args, 'encoder_embed_dim', None)
            if kdim:
                kdim += self.augment_dims * args.decoder_attention_heads
                if self.time_dependent:
                    kdim += 1 * args.decoder_attention_heads
            vdim = getattr(args, 'encoder_embed_dim', None)
            if vdim:
                vdim += self.augment_dims * args.decoder_attention_heads
                if self.time_dependent:
                    vdim += 1 * args.decoder_attention_heads
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=kdim,
                vdim=vdim,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        #self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.initial_embed_dim)

        self.final_layer_norm = LayerNorm(self.initial_embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False
        self.nfe = 0
        
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        t,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        self.nfe += 1
        if self.time_dependent:
            # enhancing x with time
            # Shape (seq_len, batch_size, 1)
            t_vec = torch.ones(x.shape[0], x.shape[1], 1 * self.decoder_attention_heads).to(x) * t
            # Shape (seq_len, batch_size, data_dim + 1)
            x = torch.cat([t_vec, x], 2)
            # enhancing encoder_out with time
            t_vec = torch.ones(encoder_out.shape[0],encoder_out.shape[1], 1 * self.decoder_attention_heads).to(encoder_out) * t
            encoder_out = torch.cat([t_vec, encoder_out], 2)
        #residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        #x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            #residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            #x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        #residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        # customizing a bit because with ODE we can't return many values for ODE Solver
        self.cur_attn = attn
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            #return x, attn, self_attn_state
            # customizing a bit because with ODE we can't return many values for ODE Solver
            self.cur_self_attn_state = self_attn_state
            return x
        #return x, attn
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn



class NodeTransformerDecoderLayer_Separated(nn.Module):
    """Separated decoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.method = args.node_method
        self.rtol = args.node_rtol
        self.atol = args.node_atol        
        self.func_self_attn = TransformerDecoderLayerFunc_SelfAttn(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.func_ode = NodeTransformerDecoderLayerFunc_EncoderAttn(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.ts = torch.FloatTensor(args.node_ts)
        self.max_num_steps = args.node_max_num_steps
        self.augment_dims = args.node_augment_dims
        self.time_dependent = args.node_time_dependent
        self.decoder_attention_heads = args.decoder_attention_heads
        

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):

        # not used for now
        # output = odeint(self.func1, x, self.ts,
        #                method=self.method,
        #                options={
        #                    "incremental_state": incremental_state,
        #                    "prev_self_attn_state": prev_self_attn_state,
        #                    "prev_attn_state": prev_attn_state,
        #                    "self_attn_mask": self_attn_mask,
        #                    "self_attn_padding_mask": self_attn_padding_mask,
        #                    #'max_num_steps': self.max_num_steps,
        #                },
        #                rtol=self.rtol, atol=self.atol)
        # keep only last time step
        # x = output[-1, :, :]
        x, self_attn = self.func_self_attn(x, incremental_state, prev_self_attn_state, prev_attn_state,
                  self_attn_mask, self_attn_padding_mask)
        
        # augment for NODE
        if self.augment_dims > 0:
            seq_len, batch, embed_dim = x.shape
            aug = torch.zeros(seq_len, batch, self.augment_dims * self.decoder_attention_heads).to(x)
            # Shape (seq_len, batch, embed_dim + augment_dims)
            x = torch.cat([x, aug], 2)
            
            seq_len, batch, embed_dim = encoder_out.shape
            aug = torch.zeros(seq_len, batch, self.augment_dims * self.decoder_attention_heads).to(encoder_out)
            # Shape (seq_len, batch, embed_dim + augment_dims)
            encoder_out = torch.cat([encoder_out, aug], 2)
            
        output = odeint(self.func_ode, x, self.ts,
                        method=self.method,
                        options={
                            "encoder_out": encoder_out,
                            "encoder_padding_mask": encoder_padding_mask,
                            "incremental_state": incremental_state,
                            "prev_self_attn_state": prev_self_attn_state,
                            "prev_attn_state": prev_attn_state,
                            "self_attn_mask": self_attn_mask,
                            "self_attn_padding_mask": self_attn_padding_mask,
                            #'max_num_steps': self.max_num_steps,
                        },
                        rtol=self.rtol, atol=self.atol)
        # keep only last time step
        output = output[-1, :, :]
        # customizing a bit because with ODE we can't return many values for ODE Solver
        if self.func_ode.onnx_trace and incremental_state is not None:
            return output, self.func_ode.cur_attn, self_attn
        else:
            return output, self_attn

    def nfe(self):
        return self.func_ode.nfe
    
    def reset_nfe(self):
        self.func_ode.nfe = 0    
            
        
class TransformerDecoderLayerFunc_SelfAttn(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.decoder_attention_heads = args.decoder_attention_heads
        
        self.embed_dim = args.decoder_embed_dim
            
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )
        self.dropout = args.dropout

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.normalize_before = args.decoder_normalize_before

        self.need_attn = True

        self.onnx_trace = False
        self.nfe = 0
        
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        # customizing a bit because with ODE we can't return many values for ODE Solver
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn
        
        

class NodeTransformerDecoderLayerFunc_SelfAttn(nn.Module):
    """Node Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.augment_dims = args.node_augment_dims
        self.time_dependent = args.node_time_dependent
        self.decoder_attention_heads = args.decoder_attention_heads
        
        self.embed_dim = args.decoder_embed_dim + self.augment_dims * args.decoder_attention_heads
        self.initial_embed_dim = self.embed_dim
        if self.time_dependent:
            self.embed_dim += 1 * args.decoder_attention_heads
            
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )
        self.dropout = args.dropout

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.normalize_before = args.decoder_normalize_before

        self.need_attn = True

        self.onnx_trace = False
        self.nfe = 0
        
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        t,
        x,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        self.nfe += 1
        if self.time_dependent:
            # enhancing x with time
            # Shape (seq_len, batch_size, 1)
            t_vec = torch.ones(x.shape[0], x.shape[1], 1 * self.decoder_attention_heads).to(x) * t
            # Shape (seq_len, batch_size, data_dim + 1)
            x = torch.cat([t_vec, x], 2)
        #residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        #x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        # customizing a bit because with ODE we can't return many values for ODE Solver
        self.cur_attn = attn
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            #return x, attn, self_attn_state
            # customizing a bit because with ODE we can't return many values for ODE Solver
            self.cur_self_attn_state = self_attn_state
            return x
        #return x, attn
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

        

class NodeTransformerDecoderLayerFunc_EncoderAttn(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.augment_dims = args.node_augment_dims
        self.time_dependent = args.node_time_dependent
        self.decoder_attention_heads = args.decoder_attention_heads
        
        self.embed_dim = args.decoder_embed_dim + self.augment_dims * args.decoder_attention_heads
        self.initial_embed_dim = self.embed_dim
        if self.time_dependent:
            self.embed_dim += 1 * args.decoder_attention_heads
            
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            kdim = getattr(args, 'encoder_embed_dim', None)
            if kdim:
                kdim += self.augment_dims * args.decoder_attention_heads
                if self.time_dependent:
                    kdim += 1 * args.decoder_attention_heads
            vdim = getattr(args, 'encoder_embed_dim', None)
            if vdim:
                vdim += self.augment_dims * args.decoder_attention_heads
                if self.time_dependent:
                    vdim += 1 * args.decoder_attention_heads
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=kdim,
                vdim=vdim,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        #self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.initial_embed_dim)

        self.final_layer_norm = LayerNorm(self.initial_embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False
        self.nfe = 0
        
    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        t,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        self.nfe += 1
        if self.time_dependent:
            # enhancing x with time
            # Shape (seq_len, batch_size, 1)
            t_vec = torch.ones(x.shape[0], x.shape[1], 1 * self.decoder_attention_heads).to(x) * t
            # Shape (seq_len, batch_size, data_dim + 1)
            x = torch.cat([t_vec, x], 2)
            # enhancing encoder_out with time
            t_vec = torch.ones(encoder_out.shape[0],encoder_out.shape[1], 1 * self.decoder_attention_heads).to(encoder_out) * t
            encoder_out = torch.cat([t_vec, encoder_out], 2)
        #residual = x

        if self.encoder_attn is not None:
            #residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            #x = F.dropout(x, p=self.dropout, training=self.training)
            #x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        #residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        #x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        # customizing a bit because with ODE we can't return many values for ODE Solver
        self.cur_attn = attn

        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

        
        
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


#@register_model_architecture('node_transformer', 'node_transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    #args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    #args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    
    # NODE
    args.node_encoder = getattr(args, 'node_encoder', False)
    
    if args.node_encoder:
        args.encoder_layers = getattr(args, 'encoder_layers', 1)
    else:
        args.encoder_layers = getattr(args, 'encoder_layers', 6)
        
    args.node_decoder = getattr(args, 'node_decoder', False)
    
    if args.node_decoder:
        args.decoder_layers = getattr(args, 'decoder_layers', 1)
    else:
        args.decoder_layers = getattr(args, 'decoder_layers', 6)
    
    args.node_rtol = getattr(args, 'node_rtol', 0.01)
    args.node_atol = getattr(args, 'node_atol', 0.01)
    args.node_method = getattr(args, 'node_method', "dopri5-ext")
    args.node_max_num_steps = getattr(args, 'node_max_num_steps', DEFAULT_MAX_NUM_STEPS)
    args.node_augment_dims = getattr(args, 'node_augment_dims', DEFAULT_AUGMENT_DIMS)
    args.node_time_dependent = getattr(args, 'node_time_dependent', False)
    args.node_separated_decoder = getattr(args, 'node_separated_decoder', False)
