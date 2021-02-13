# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_bert import (BertModel, BertPreTrainedModel, BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)


class BertForDST(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForDST, self).__init__(config)
        self.slot_list = config.dst_slot_list
        self.domain_list = config.dst_domain_list
        self.class_types = config.dst_class_types
        self.class_labels = config.dst_class_labels
        self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
        self.refer_loss_for_nonpointable = config.dst_refer_loss_for_nonpointable
        self.class_aux_feats_inform = config.dst_class_aux_feats_inform
        self.class_aux_feats_ds = config.dst_class_aux_feats_ds
        self.class_loss_ratio = config.dst_class_loss_ratio

        # Only use refer loss if refer class is present in dataset.
        if 'refer' in self.class_types:
            self.refer_index = self.class_types.index('refer')
        else:
            self.refer_index = -1

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)
        
        # node embedding table
        self.node_table = NodeTable(config)
        
        # node representation
        self.graph = GraphLayer()
        
        # dialogue attention
        self.dialogue_attention = DialogueAtt()
        
        # add edge
        self.add_edge = AddEdge(config)

        if self.class_aux_feats_inform:
            self.add_module("inform_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))
        if self.class_aux_feats_ds:
            self.add_module("ds_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))

        aux_dims = len(self.slot_list) * (self.class_aux_feats_inform + self.class_aux_feats_ds) # second term is 0, 1 or 2

        for slot in self.slot_list:
            #self.add_module("class_" + slot, nn.Linear(config.hidden_size + aux_dims, self.class_labels))
            self.add_module("class_" + slot, nn.Linear(config.hidden_size + aux_dims + config.hidden_size, self.class_labels))
            self.add_module("token_" + slot, nn.Linear(config.hidden_size + config.hidden_size, 2))
            self.add_module("refer_" + slot, nn.Linear(config.hidden_size + aux_dims + config.hidden_size, len(self.slot_list) + 1))

        self.init_weights()

    def forward(self,
                input_ids,
                input_mask=None,
                segment_ids=None,
                position_ids=None,
                head_mask=None,
                start_pos=None,
                end_pos=None,
                inform_slot_id=None,
                refer_id=None,
                class_label_id=None,
                diag_state=None,
                initial_node_matrix=None,
                slot_id=None):
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # TODO: establish proper format in labels already?
        if inform_slot_id is not None:
            inform_labels = torch.stack(list(inform_slot_id.values()), 1).float()
        if diag_state is not None:
            diag_state_labels = torch.clamp(torch.stack(list(diag_state.values()), 1).float(), 0.0, 1.0)

        if self.class_aux_feats_inform and self.class_aux_feats_ds:
            pooled_output_aux = torch.cat(
                (pooled_output, self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)
        elif self.class_aux_feats_inform:
            pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)
        elif self.class_aux_feats_ds:
            pooled_output_aux = torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)
        else:
            pooled_output_aux = pooled_output
            
        # schema graph
        node_embedding_table = self.node_table()
        batch_node_embedding = node_embedding_table(slot_id)
        # real_batch_size = self.config.dst_batch_size
        # tile_node_embedding_table = node_embedding_table.repeat(real_batch_size, 1, 1)
        initial_node_embedding = self.graph(batch_node_embedding, initial_node_matrix)
        dialogue_aware_node_embedding = self.dialogue_attention(sequence_output, initial_node_embedding)
        schema_graph_structure = self.add_edge(dialogue_aware_node_embedding, initial_node_matrix)
        new_node_embedding = self.graph(dialogue_aware_node_embedding, schema_graph_structure)

        total_loss = 0
        per_slot_per_example_loss = {}
        per_slot_class_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_refer_logits = {}
        for slot_id, slot in enumerate(self.slot_list):
            # if self.class_aux_feats_inform and self.class_aux_feats_ds:
            #     pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)
            # elif self.class_aux_feats_inform:
            #     pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)
            # elif self.class_aux_feats_ds:
            #     pooled_output_aux = torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)
            # else:
            #     pooled_output_aux = pooled_output
            
            slot_embedding = new_node_embedding[:, len(self.domain_list) + slot_id, :]  # batch, dim
            # class_logits = self.dropout_heads(getattr(self, 'class_' + slot)(pooled_output_aux))
            class_logits = self.dropout_heads(getattr(self, 'class_' + slot)(torch.cat((pooled_output_aux, slot_embedding), 1)))

            slot_embedding_token = torch.reshape(slot_embedding.repeat(1, self.config.dst_sequence_len), (-1, self.config.dst_sequence_len, self.config.hidden_size))
            token_logits = self.dropout_heads(getattr(self, 'token_' + slot)(torch.cat((sequence_output, slot_embedding_token), 2)))
            start_logits, end_logits = token_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            refer_logits = self.dropout_heads(getattr(self, 'refer_' + slot)(torch.cat((pooled_output_aux, slot_embedding), 1)))

            per_slot_class_logits[slot] = class_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits
            per_slot_refer_logits[slot] = refer_logits
            
            # If there are no labels, don't compute loss
            if class_label_id is not None and start_pos is not None and end_pos is not None and refer_id is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_pos[slot].size()) > 1:
                    start_pos[slot] = start_pos[slot].squeeze(-1)
                if len(end_pos[slot].size()) > 1:
                    end_pos[slot] = end_pos[slot].squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1) # This is a single index
                start_pos[slot].clamp_(0, ignored_index)
                end_pos[slot].clamp_(0, ignored_index)

                class_loss_fct = CrossEntropyLoss(reduction='none')
                token_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)
                refer_loss_fct = CrossEntropyLoss(reduction='none')

                start_loss = token_loss_fct(start_logits, start_pos[slot])
                end_loss = token_loss_fct(end_logits, end_pos[slot])
                token_loss = (start_loss + end_loss) / 2.0

                token_is_pointable = (start_pos[slot] > 0).float()
                if not self.token_loss_for_nonpointable:
                    token_loss *= token_is_pointable

                refer_loss = refer_loss_fct(refer_logits, refer_id[slot])
                token_is_referrable = torch.eq(class_label_id[slot], self.refer_index).float()
                if not self.refer_loss_for_nonpointable:
                    refer_loss *= token_is_referrable
                print(class_logits.size)
                print(class_logits.cpu().numpy())
                class_loss = class_loss_fct(class_logits, class_label_id[slot])

                if self.refer_index > -1:
                    per_example_loss = (self.class_loss_ratio) * class_loss + ((1 - self.class_loss_ratio) / 2) * token_loss + ((1 - self.class_loss_ratio) / 2) * refer_loss
                else:
                    per_example_loss = self.class_loss_ratio * class_loss + (1 - self.class_loss_ratio) * token_loss

                total_loss += per_example_loss.sum()
                per_slot_per_example_loss[slot] = per_example_loss

        # add hidden states and attention if they are here
        outputs = (total_loss,) + (per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits,) + outputs[2:]

        return outputs
    

class NodeTable(nn.Module):
    
    def __init__(self, config):
        super(NodeTable, self).__init__()
        self.domain_num = len(config.dst_domain_list)
        self.slot_num = len(config.dst_slot_list)
        self.node_num = self.domain_num + self.slot_num
        self.node_table = nn.Embedding(self.node_num, config.hidden_size)

    def forward(self):
        return self.node_table


class GraphLayer(nn.Module):
    
    def __init__(self, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)
        
    def forward(self, node_embedding, adjacent_matrix):
        x = torch.matmul(node_embedding, torch.transpose(node_embedding, 1, 2))
        x[~adjacent_matrix] = float(-10000000)
        att = torch.softmax(x, dim=2)
        w_node_embedding = att.matmul(node_embedding)
        return w_node_embedding
    

class DialogueAtt(nn.Module):
    
    def __init__(self, **kwargs):
        super(DialogueAtt, self).__init__(**kwargs)
        
    def forward(self, token, node):
        att = torch.softmax(node.matmul(torch.transpose(token, 1, 2)), dim=2)
        w_node_embedding = att.matmul(token)
        return w_node_embedding


class AddEdge(nn.Module):
    
    def __init__(self, config):
        super(AddEdge, self).__init__()
        self.domain_num = len(config.dst_domain_list)
        self.slot_num = len(config.dst_slot_list)
        self.node_num = self.domain_num + self.slot_num
        self.params = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))

    def forward(self, node, node_matrix):
        prob = torch.sigmoid(node.matmul(self.params).matmul(torch.transpose(node, 1, 2)))
        pred = (prob > 0.5).float()
        slot_slot_pred = pred[:, self.domain_num:, self.domain_num:]
        slot_slot_initial = node_matrix[:, self.domain_num:, self.domain_num:]
        slot_slot = ((slot_slot_pred + slot_slot_initial) >= 1).long()
        domain_domain = node_matrix[:, :self.domain_num, :self.domain_num]
        domain_slot = node_matrix[:, :self.domain_num, self.domain_num:]
        slot_domain = node_matrix[:, self.domain_num:, :self.domain_num]
        new_edge = torch.cat((torch.cat((domain_domain, domain_slot), 2), torch.cat((slot_domain, slot_slot), 2)), 1)
        return new_edge
