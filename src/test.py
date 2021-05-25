import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from metric import Metric
from models import SpellBertPho2ResArch3
from models_abla import SpellBertPho2ResArch3Abla
from utils import Pinyin2

pho2_convertor = Pinyin2()


MODEL_CLASSES = {  
    'bert-pho2-res-arch3': SpellBertPho2ResArch3,
    'bert-pho2-res-arch3-abla': SpellBertPho2ResArch3Abla,
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

    bs = 32
    batches = []
    r = len(dataset)
    for i in tqdm(range(0, len(dataset), bs)):
        batches.append(make_features(
            dataset[i:min(i+bs,r)],
            tokenizer,
            batch_processor=build_batch,
        ))

    return batches


def test(ckpt_dir, data_dir, ckpt_num, testset_year, output_dir, device):
    weight_dir = ckpt_dir
    if ckpt_num == -1:
        ckpt_name = f'last'
        model_dir = weight_dir
    else:
        ckpt_name = f'saved_ckpt-{ckpt_num}'
        model_dir = os.path.join(weight_dir, ckpt_name)

    if testset_year == 13:
        test_picke_path = os.path.join(data_dir, 'test.sighan13.pkl')
        label_path = os.path.join(data_dir, 'test.sighan13.lbl.tsv')
    elif testset_year == 14:
        test_picke_path = os.path.join(data_dir, 'test.sighan14.pkl')
        label_path = os.path.join(data_dir, 'test.sighan14.lbl.tsv')
    elif testset_year == 15:
        test_picke_path = os.path.join(data_dir, 'test.sighan15.pkl')
        label_path = os.path.join(data_dir, 'test.sighan15.lbl.tsv')
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
    device = torch.device(device)

    # Model
    print('Load model begin...')
    model = model_class.from_pretrained(model_dir)
    model = model.to(device)
    model = model.eval()
    print('Load model done.')

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

    # Metric
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pred_txt_path = os.path.join(output_dir, 'preds.txt')
    pred_lbl_path = os.path.join(output_dir, 'labels.txt')
    metric = Metric(vocab_path=weight_dir)
    results = metric.metric(
        batches=batches,
        pred_txt_path=pred_txt_path,
        pred_lbl_path=pred_lbl_path,
        label_path=label_path,
        should_remove_de=testset_year == 13,
    )
    for key in sorted(results.keys()):
        print(f'{key}: {results[key]:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--testset_year', type=int, required=True, choices=[15, 14, 13])
    parser.add_argument('--ckpt_num', type=int, default=-1)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default="cuda:0")
    args = parser.parse_args()

    test(
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
        ckpt_num=args.ckpt_num,
        testset_year=args.testset_year,
        output_dir=args.output_dir,
        device=args.device,
    )
