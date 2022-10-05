# torch related
import sys
sys.path.append('.')
sys.path.append('..')
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from network.pointnet_encoder import PointNetEncoder
from network.CVAE import VAE
from src.modeling.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertPooler, BertIntermediate, BertOutput, BertSelfOutput
import src.modeling.data.config as cfg
from src.modeling.bert.modeling_utils import prune_linear_layer

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hiddensize (%d) is not a multiple of the number of attention heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.output_attentions = config.output_attentions
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_layer = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None, history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query = self.query_layer(hidden_states)
            mixed_key = self.key_layer(x_states)
            mixed_value = self.value_layer(x_states)

        else:
            mixed_query = self.query_layer(hidden_states)
            mixed_key = self.key_layer(hidden_states)
            mixed_value = self.value_layer(hidden_states)

        query = self.transpose_for_scores(mixed_query)
        key = self.transpose_for_scores(mixed_key)
        value = self.transpose_for_scores(mixed_value)

        # Take dot product between 'query' and 'key' to get the raw attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask # the attention mask is precomputed for all layers in BertModel forward() function

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else(context_layer,)
        return outputs

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self_att = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self_att.num_attention_heads, self.self_att.attention_head_size)
        for head in heads:
            mask[head] = 0 # set all the params of the selected head to 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self_att.query_layer = prune_linear_layer(self.self_att.query_layer, index)
        self.self_att.key_layer = prune_linear_layer(self.self_att.key_layer, index)
        self.self_att.value_layer = prune_linear_layer(self.self_att.value_layer, index)

        self.output.dense = prune_linear_layer(self.output, index)

        # Update hyper params
        self.self_att.num_attention_heads = self.self_att.num_attention_heads - len(heads)
        self.self_att.all_head_size = self.self_att.attention_head_size * self.self_att.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None, history_state=None):
        self_outputs = self.self_att(input_tensor, attention_mask, head_mask, history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:] # add attention

        return outputs

class ObjContactormerLayer(nn.Module):
    def __init__(self, config):
        super(ObjContactormerLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None, history_state=None):
        return

class ObjContactormerEncoder(nn.Module):
    def __init__(self, config):
        super(ObjContactormerEncoder, self).__init__()
        self.output_attention = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([ObjContactormerLayer(config) for _ in range(config.num_hidden_layers)]) # no need of particular loop var

    def forward(self, hidden_states, attention_mask, head_mask=None, encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            history_state = None if encoder_history_states is None else encoder_history_states[i]

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], history_state)
            hidden_states = layer_outputs[0] # update hidden states

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer: no other operations, just output current hidden_states
        outputs = (hidden_states,)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)


        return outputs # outputs, (hidden states), (attentions)

class EncoderBlock(BertPreTrainedModel):
    def __init__(self, config):
        super(EncoderBlock, self).__init__(config)
        self.cnofig = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = ObjContactormerEncoder(config)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.feat_dim = config.img_feature_dim

        try: self.use_feat_layernorm = config.use_feat_layernorm # use_img_layernorm -> use_feat_layernorm
        except: self.use_feat_layernorm = None

        self.feat_embedding = nn.Linear(self.feat_dim, self.config.hidden_size, bias=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)
        self.apply(self.init_weights) # init_weights from inherited BertPreTrainedModel

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        return
        
        

class ObjContactormer(nn.Module):
    """
    
    """
    def __init__(self, in_channel_obj=4, in_channel_hand=3, encoder_sizes=[1024, 512, 256], \
                latent_size=64, decoder_sizes=[1024, 256, 61], condition_size=1024):
        super(ObjContactormer).__init__()

        self.in_channel_obj = in_channel_obj

    def forward(self, obj_pc, hand_xyz):
        """
        """

        B = obj_pc.size(0)
        
        return 