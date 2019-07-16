from fairseq.models import register_model_architecture
from odeint_ext.odeint_ext import odeint_adjoint_ext as odeint
#from node_transformer import node_transformer
from node_transformer.node_transformer import base_architecture


@register_model_architecture('node_transformer', 'node_transformer')
def node_transformer(args):
    base_architecture(args)
    
    
@register_model_architecture('node_transformer', 'node_transformer_wmt_en_fr')
def node_transformer_wmt_en_fr(args):
    base_architecture(args)
