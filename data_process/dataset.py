import argparse
import pickle
import random

import torch
import transformers
from tqdm import tqdm

from phonetics.phonetics import Phonetics


def data_to_pickle(data_path, pickle_path, vocab_path, sort_item,
                   shuffle_batch, max_len, batch_size,
                   phonetics_vocab_dir, with_phonetics):
    dataset = build_dataset(
        data_path=data_path,
        vocab_path=vocab_path,
        max_len=max_len,
        phonetics_vocab_dir=phonetics_vocab_dir,
        with_phonetics=with_phonetics,
    )
    build_batch(
        dataset=dataset,
        pickle_path=pickle_path,
        sort_item=sort_item,
        shuffle_batch=shuffle_batch,
        batch_size=batch_size,
    )


def build_dataset(data_path, vocab_path, max_len, phonetics_vocab_dir, with_phonetics):
    # Load Data
    data_raw = []
    with open(data_path, encoding='utf8') as f:
        data_raw = [s.split('\t') for s in f.read().splitlines()]
    print(f'#Item: {len(data_raw)} from "{data_path}"')

    # Vocab
    tokenizer = transformers.BertTokenizer(vocab_path)

    # Phonetics vocab
    if with_phonetics == 'yes':
        phonetics = Phonetics(vocab_dir=phonetics_vocab_dir)
    else:
        phonetics = None

    # Data Basic
    data = []
    for item_raw in tqdm(data_raw, desc='Build Dataset'):
        # Field: id, src, tgt
        item = {
            'id': item_raw[0],
            'src': item_raw[1],
            'tgt': item_raw[2],
        }
        assert len(item['src']) == len(item['tgt'])
        data.append(item)

        # Field: tokens_size
        tokens = tokenizer.tokenize(item['src'])
        tokens_size = []
        for t in tokens:
            if t == tokenizer.unk_token:
                tokens_size.append(1)
            elif t.startswith('##'):
                tokens_size.append(len(t) - 2)
            else:
                tokens_size.append(len(t))
        item['tokens_size'] = tokens_size

        # Field: src_idx
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        item['src_idx'] = ids

        # Field: tgt_idx
        item['tgt_idx'] = tokenizer.encode(item['tgt'])
        assert len(item['src_idx']) == len(item['tgt_idx'])

        # Field: labels
        # item['labels'] = [int(a != b) for a, b in zip(item['src_idx'], item['tgt_idx'])]
        # assert len(item['labels']) == len(item['src_idx'])
        
    # Phonetic: Field: src_consonant_idx, src_vowel_idx, src_tone_idx
    if with_phonetics == 'yes':
        for item in tqdm(data, desc='Add Phonetics'):
            tokens = tokenizer.tokenize(item['src'])
            item['src_consonant_idx'], item['src_vowel_idx'], item['src_tone_idx'] = \
                phonetics.convert_tokens_to_phonetic_idxs(tokens, add_cls_sep=True)

            assert len(item['src_idx']) == len(item['src_consonant_idx'])
            assert len(item['src_idx']) == len(item['src_vowel_idx'])
            assert len(item['src_idx']) == len(item['src_tone_idx'])

    # Trim
    if max_len > 0:
        n_all_items = len(data)
        data = [item for item in data if len(item['src_idx']) <= max_len]
        n_filter_items = len(data)
        n_cut = n_all_items - n_filter_items
        print(f'max_len={max_len}, {n_all_items} -> {n_filter_items} ({n_cut})')

    return data


def build_batch(dataset, pickle_path, sort_item, shuffle_batch, batch_size):
    if sort_item:
        dataset.sort(key=lambda d: len(d['src_idx']))

    class BatchSampler:
        
        def __init__(self, n_item, batch_size, shuffle):
            self.shuffle = shuffle
            self.idxs_list = []
            idxs = []
            for i in range(n_item):
                idxs.append(i)
                if len(idxs) == batch_size:
                    self.idxs_list.append(idxs)
                    idxs = []
            if len(idxs) > 0:
                self.idxs_list.append(idxs)

        def __iter__(self):
            if self.shuffle:
                random.shuffle(self.idxs_list)
            for idxs in self.idxs_list:
                yield idxs

        def __len__(self):
            return len(self.idxs_list)

    def collate_fn(items):
        # Field: src, tgt
        batch = {}
        for k in items[0]:
            batch[k] = [item[k] for item in items]

        def pad(seq):
            return torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x) for x in seq],
                batch_first=True,
                padding_value=0,
            )

        # Field: src_idx, tgt_idx, labels
        batch['src_idx'] = pad(batch['src_idx'])
        batch['tgt_idx'] = pad(batch['tgt_idx'])
        # batch['labels'] = pad(batch['labels'])

        # Field: src_consonant_idx, src_vowel_idx, src_tone_idx
        if 'src_consonant_idx' in batch:
            batch['src_consonant_idx'] = pad(batch['src_consonant_idx'])
            batch['src_vowel_idx'] = pad(batch['src_vowel_idx'])
            batch['src_tone_idx'] = pad(batch['src_tone_idx'])

        # Field: lengths
        lens = [len(item['src_idx']) - 2 for item in items]  # ignore [CLS & [SEP]
        batch['lengths'] = lens

        return batch

    batch_sampler = BatchSampler(
        n_item=len(dataset),
        batch_size=batch_size,
        shuffle=shuffle_batch,
    )
    batches = []
    for idxs in batch_sampler:
        items = [dataset[i] for i in idxs]
        batch = collate_fn(items)
        batches.append(batch)
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(batches, f)


def build_dataloader(pickle_path, shuffle):
    with open(pickle_path, 'rb') as f:
        batches = pickle.load(f)
    if shuffle:
        random.shuffle(batches)
    loader = torch.utils.data.DataLoader(
        dataset=batches,
        batch_size=None,
    )
    return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--pickle_path', required=True)
    parser.add_argument('--vocab_path', default='/data/dobby_ceph_ir/hengdaxu/.local/chinese_wwm_ext/vocab.txt')
    
    parser.add_argument('--sort_item', choices=['yes', 'no'], required=True)
    parser.add_argument('--shuffle_batch', choices=['yes', 'no'], required=True)
    parser.add_argument('--max_len', type=int, required=True)
    parser.add_argument('--batch_size', default=32, type=int, required=True)

    parser.add_argument('--with_phonetics', choices=['yes', 'no'], required=True)
    parser.add_argument('--phonetics_vocab_dir', default='phonetics/vocab')
    args = parser.parse_args()

    data_to_pickle(
        data_path=args.data_path,
        pickle_path=args.pickle_path,
        vocab_path=args.vocab_path,
        sort_item=True if args.sort_item == 'yes' else False,
        shuffle_batch=True if args.shuffle_batch == 'yes' else False,
        max_len=args.max_len,
        batch_size=args.batch_size,
        phonetics_vocab_dir=args.phonetics_vocab_dir,
        with_phonetics=args.with_phonetics,
    )
