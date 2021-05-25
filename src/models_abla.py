from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import os
import sys

import opencc
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_bert import *
from utils import pho_convertor, pho2_convertor
from copy import deepcopy
from PIL import ImageFont
import numpy as np

from char_cnn import CharResNet, CharResNet1

def _is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


class SpellBertPho2ResArch3Abla(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.config.fusion = getattr(config, 'fusion', 'gate')
        self.config.with_pho = getattr(config, 'with_pho', 'yes')
        self.config.with_res = getattr(config, 'with_res', 'yes')
        self.config.num_gates = 1
        if self.config.with_pho == 'yes':
            self.config.num_gates += 1
        if self.config.with_res == 'yes':
            self.config.num_gates += 1

        # Semantic Encoder
        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        # Prounciation Encoder
        if self.config.with_pho == 'yes':
            self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
            self.pho_gru = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0,
                bidirectional=False,
            )
            pho_config = deepcopy(config)
            pho_config.num_hidden_layers = 4
            self.pho_model = BertModel(pho_config)

        # Vision Encoder
        if self.config.with_res == 'yes':
            if self.config.num_fonts == 1:
                self.char_images = nn.Embedding(config.vocab_size, 1024)
                self.char_images.weight.requires_grad = False
            else:
                self.char_images_multifonts = torch.nn.Parameter(torch.rand(21128, self.config.num_fonts, 32, 32))
                self.char_images_multifonts.requires_grad = False

            if config.image_model_type == 0:
                self.resnet = CharResNet(in_channels=self.config.num_fonts)
            elif config.image_model_type == 1:
                self.resnet = CharResNet1()
            else:
                raise NotImplementedError('invalid image_model_type %d'%config.image_model_type)
            self.resnet_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Fusion
        if self.config.fusion == 'gate':
            self.gate_net = nn.Linear((self.config.num_gates + 1) * config.hidden_size, self.config.num_gates)
        print('gate:', self.config.fusion)

        out_config = deepcopy(config)
        out_config.num_hidden_layers = 3
        self.output_block = BertModel(out_config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    def build_glyce_embed_multifonts(self, vocab_dir, num_fonts, use_traditional_font, font_size=32):
        font_paths = [
            ('simhei.ttf', False),
            ('xiaozhuan.ttf', False),
            ('simhei.ttf', True),
        ]
        font_paths = font_paths[:num_fonts]
        if use_traditional_font:
            font_paths = font_paths[:-1]
            font_paths.append(('simhei.ttf', True))
            self.converter = opencc.OpenCC('s2t.json')

        images_list = []
        for font_path, use_traditional in font_paths:
            images = self.build_glyce_embed_onefont(
                vocab_dir=vocab_dir,
                font_path=font_path,
                font_size=font_size,
                use_traditional=use_traditional,
            )
            images_list.append(images)

        char_images = torch.stack(images_list, dim=1).contiguous()
        self.char_images_multifonts.data.copy_(char_images)

    def build_glyce_embed_onefont(self, vocab_dir, font_path, font_size, use_traditional):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path) as f:
            vocab = [s.strip() for s in f.readlines()]
        if use_traditional:
            vocab = [self.converter.convert(c) if len(c) == 1 else c for c in vocab]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) > 1:
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).contiguous()
        return char_images

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['src_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

        input_shape = input_ids.size()

        # Semantic Encoder
        bert_hiddens = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        # Prounciation Encoder
        if self.config.with_pho == 'yes':
            pho_embeddings = self.pho_embeddings(pho_idx)
            pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
                input=pho_embeddings,
                lengths=pho_lens,
                batch_first=True,
                enforce_sorted=False,
            )
            _, pho_hiddens = self.pho_gru(pho_embeddings)
            pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
            pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens, attention_mask=attention_mask)[0]
        else:
            pho_hiddens = None

        # Vision Encoder
        if self.config.with_res == 'yes':
            src_idxs = input_ids.view(-1)

            if self.config.num_fonts == 1:
                images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
            else:
                images = self.char_images_multifonts.index_select(dim=0, index=src_idxs)

            res_hiddens = self.resnet(images)
            res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()
            res_hiddens = self.resnet_layernorm(res_hiddens)
        else:
            res_hiddens = None

        # Fusion
        if self.config.fusion == 'gate':
            bert_hiddens_mean = (bert_hiddens * attention_mask.to(torch.float).unsqueeze(2)).sum(dim=1) / attention_mask.to(torch.float).sum(dim=1, keepdim=True)
            bert_hiddens_mean = bert_hiddens_mean.unsqueeze(1).expand(-1, bert_hiddens.size(1), -1)

            if self.config.with_pho == 'yes' and self.config.with_res == 'yes':
                modal_states = [bert_hiddens, pho_hiddens, res_hiddens, bert_hiddens_mean]
                concated_outputs = torch.cat(modal_states, dim=-1)
                gated_values = self.gate_net(concated_outputs)
                g0 = torch.sigmoid(gated_values[:,:,0].unsqueeze(-1))
                g1 = torch.sigmoid(gated_values[:,:,1].unsqueeze(-1))
                g2 = torch.sigmoid(gated_values[:,:,2].unsqueeze(-1))
                hiddens = g0 * bert_hiddens + g1 * pho_hiddens + g2 * res_hiddens
            elif self.config.with_pho == 'yes' and self.config.with_res == 'no':
                modal_states = [bert_hiddens, pho_hiddens, bert_hiddens_mean]
                concated_outputs = torch.cat(modal_states, dim=-1)
                gated_values = self.gate_net(concated_outputs)
                g0 = torch.sigmoid(gated_values[:,:,0].unsqueeze(-1))
                g1 = torch.sigmoid(gated_values[:,:,1].unsqueeze(-1))
                hiddens = g0 * bert_hiddens + g1 * pho_hiddens
            elif self.config.with_pho == 'no' and self.config.with_res == 'yes':
                modal_states = [bert_hiddens, res_hiddens, bert_hiddens_mean]
                concated_outputs = torch.cat(modal_states, dim=-1)
                gated_values = self.gate_net(concated_outputs)
                g0 = torch.sigmoid(gated_values[:,:,0].unsqueeze(-1))
                g2 = torch.sigmoid(gated_values[:,:,1].unsqueeze(-1))
                hiddens = g0 * bert_hiddens + g2 * res_hiddens
            else:
                modal_states = [bert_hiddens, bert_hiddens_mean]
                concated_outputs = torch.cat(modal_states, dim=-1)
                gated_values = self.gate_net(concated_outputs)
                g0 = torch.sigmoid(gated_values[:,:,0].unsqueeze(-1))
                hiddens = g0 * bert_hiddens
        else:
            hiddens = bert_hiddens + pho_hiddens + res_hiddens

        outputs = self.output_block(
            inputs_embeds=hiddens,
            position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs
        return outputs 
