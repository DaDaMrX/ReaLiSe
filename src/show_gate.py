import argparse
import os
import pickle
import types

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from models import SpellBertPho2ResArch3
from utils import Pinyin2
from torch.nn import CrossEntropyLoss

pho2_convertor = Pinyin2()


MODEL_CLASSES = {  
    'bert-pho2-res-arch3': SpellBertPho2ResArch3,    
    'bert-pho2-res-arch3-mlm': SpellBertPho2ResArch3MLM,
    'bert-pho2-res-arch3-trans': SpellBertPho2ResArch3BertTrans,
    'bert-pho2-res-arch3-trans-res': SpellBertPho2ResArch3BertTransRes,
    'bert-pho2-res-arch3-res': SpellBertPho2ResArch3Res,
    'bert-pho2-res-arch3-init': SpellBertPho2ResArch3Init,
}


def build_batch(batch, tokenizer):
    src_idx = batch['src_idx'].flatten().tolist()
    chars = tokenizer.convert_ids_to_tokens(src_idx)
    pho_idx, pho_lens = pho2_convertor.convert(chars)
    batch['pho_idx'] = pho_idx
    batch['pho_lens'] = pho_lens
    return batch


def make_features(examples, tokenizer, batch_processor):
    max_length = 128
    batch = {}
    for t in ['id', 'src', 'tgt', 'tokens_size', 'lengths', 'src_idx', 'tgt_idx', 'masks', 'loss_masks']:
        batch[t] = []
    for item in examples:
        for t in item:
            if t == 'src_idx' or t == 'tgt_idx':
                seq = item[t][:max_length]
                padding_length = max_length - len(seq)
                batch[t].append(seq + ([0]*padding_length))
                if t == 'src_idx':
                    batch['masks'].append(([1]*len(seq)) + ([0]*padding_length))
            elif t == 'lengths':
                batch[t].append(item[t])
                loss_mask = [0] * max_length
                for i in range(1, min(1+item[t], max_length)):
                    loss_mask[i] = 1
                batch['loss_masks'].append(loss_mask)
            else:
                batch[t].append(item[t])

    batch['src_idx'] = torch.tensor(batch['src_idx'], dtype=torch.long)
    batch['tgt_idx'] = torch.tensor(batch['tgt_idx'], dtype=torch.long)
    batch['masks'] = torch.tensor(batch['masks'], dtype=torch.long)
    batch['loss_masks'] = torch.tensor(batch['loss_masks'], dtype=torch.long)

    batch = batch_processor(batch, tokenizer)
    return batch


def prepare_batches(test_picke_path, tokenizer_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    with open(test_picke_path, 'rb') as f:
        dataset = pickle.load(f)

    bs = 1
    batches = []
    r = len(dataset)
    for i in tqdm(range(0, len(dataset), bs)):
        batches.append(make_features(
            dataset[i:min(i+bs,r)],
            tokenizer,
            batch_processor=build_batch,
        ))

    return batches


def forward(self, batch):
    input_ids = batch['src_idx']
    attention_mask = batch['masks']
    loss_mask = batch['loss_masks']
    pho_idx = batch['pho_idx']
    pho_lens = batch['pho_lens']
    label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None

    input_shape = input_ids.size()

    bert_hiddens = self.bert(input_ids, attention_mask=attention_mask)[0]
    
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

    src_idxs = input_ids.view(-1)

    if self.config.num_fonts == 1:
        images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
    else:
        images = self.char_images_multifonts.index_select(dim=0, index=src_idxs)

    res_hiddens = self.resnet(images)
    res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()
    res_hiddens = self.resnet_layernorm(res_hiddens)

    bert_hiddens_mean = (bert_hiddens * attention_mask.to(torch.float).unsqueeze(2)).sum(dim=1) / attention_mask.to(torch.float).sum(dim=1, keepdim=True)
    bert_hiddens_mean = bert_hiddens_mean.unsqueeze(1).expand(-1, bert_hiddens.size(1), -1)

    concated_outputs = torch.cat((bert_hiddens, pho_hiddens, res_hiddens, bert_hiddens_mean), dim=-1)
    gated_values = self.gate_net(concated_outputs)
    # B * S * 3
    g0 = torch.sigmoid(gated_values[:,:,0].unsqueeze(-1))
    g1 = torch.sigmoid(gated_values[:,:,1].unsqueeze(-1))
    g2 = torch.sigmoid(gated_values[:,:,2].unsqueeze(-1))

    batch['g0'] = g0.detach().cpu().squeeze(-1)
    batch['g1'] = g1.detach().cpu().squeeze(-1)
    batch['g2'] = g2.detach().cpu().squeeze(-1)
    
    hiddens = g0* bert_hiddens + g1* pho_hiddens + g2* res_hiddens

    outputs = self.output_block(inputs_embeds=hiddens,
                position_ids=torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device),
                attention_mask=attention_mask)

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


def test(dir_name, ckpt_num, testset_year):
    weight_dir = f'/data/dobby_ceph_ir/hengdaxu/venus_outputs/{dir_name}'
    if ckpt_num == -1:
        ckpt_name = f'last'
        model_dir = weight_dir
    else:
        ckpt_name = f'saved_ckpt-{ckpt_num}'
        model_dir = os.path.join(weight_dir, ckpt_name)

    if testset_year == 13:
        test_picke_path='/data/dobby_ceph_ir/hengdaxu/spell-acl/data/test.sighan13.pkl'
        label_path='/data/dobby_ceph_ir/hengdaxu/spell-acl/data/test.sighan13.lbl.tsv'
    elif testset_year == 14:
        test_picke_path='/data/dobby_ceph_ir/hengdaxu/spell-acl-data/test.sighan14.pkl'
        label_path='/data/dobby_ceph_ir/hengdaxu/spell-acl-data/test.sighan14.lbl.tsv'
    elif testset_year == 15:
        test_picke_path='/data/dobby_ceph_ir/hengdaxu/spell-acl-data/test.sighan15.pkl'
        label_path='/data/dobby_ceph_ir/hengdaxu/spell-acl-data/test.sighan15.lbl.tsv'
    else:
        raise ValueError(f'testset_year={testset_year}')

    # model_type
    training_args = torch.load(os.path.join(weight_dir, 'training_args.bin'))
    model_type = training_args.model_type
    model_class = MODEL_CLASSES[model_type]

    # Log
    print(f'model_type: {model_type}')
    print(f'weight_dir: {weight_dir}')
    print(f'ckpt_name: {ckpt_name}')

    # Dataset
    batches = prepare_batches(
        test_picke_path=test_picke_path,
        tokenizer_path=weight_dir,
    )
    print('test_batches:', len(batches))

    # Device
    device = torch.device('cuda:0')

    # Model
    print('Load model begin...')
    model = model_class.from_pretrained(model_dir)
    model = model.to(device)
    model = model.eval()
    print('Load model done.')

    model.forward = types.MethodType(forward, model)

    # Test epoch
    for batch in tqdm(batches):
        for t in batch:
            if t not in ['id', 'src', 'tgt', 'lengths', 'tokens_size', 'pho_lens']:
                batch[t] = batch[t].to(device)

        with torch.no_grad():
            outputs = model(batch)

        logits = outputs[1]

        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=-1)
        batch['src_idx'] = batch['src_idx'].detach().cpu().numpy()
        batch['pred_idx'] = preds

    # Show gate
    tokenizer = BertTokenizer.from_pretrained(weight_dir)
    rows = []
    for batch in batches:
        src_tokens = tokenizer.convert_ids_to_tokens(batch['src_idx'][0])
        tgt_tokens = tokenizer.convert_ids_to_tokens(batch['tgt_idx'][0])
        prd_tokens = tokenizer.convert_ids_to_tokens(batch['pred_idx'][0])
        g0 = batch['g0'][0].tolist()
        g1 = batch['g1'][0].tolist()
        g2 = batch['g2'][0].tolist()
        
        cut_pos = src_tokens.index('[PAD]')

        src_tokens = src_tokens[:cut_pos]
        tgt_tokens = tgt_tokens[:cut_pos]
        prd_tokens = prd_tokens[:cut_pos]
        g0 = g0[:cut_pos]
        g1 = g1[:cut_pos]
        g2 = g2[:cut_pos]

        g0 = list(map(str, g0))
        g1 = list(map(str, g1))
        g2 = list(map(str, g2))

        rows.append('\t'.join(src_tokens))
        rows.append('\t'.join(tgt_tokens))
        rows.append('\t'.join(prd_tokens))
        rows.append('\t'.join(g0))
        rows.append('\t'.join(g1))
        rows.append('\t'.join(g2))


    with open('../gate.tsv', 'w') as f:
        f.write('\n'.join(rows))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-d', required=True)
    parser.add_argument('--ckpt_num', '-c', type=int, default=-1)
    parser.add_argument('--testset_year', '-y', type=int, default=14)
    args = parser.parse_args()

    test(
        dir_name=args.output_dir,
        ckpt_num=args.ckpt_num,
        testset_year=args.testset_year,
    )
