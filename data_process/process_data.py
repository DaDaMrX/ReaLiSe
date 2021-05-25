import sys,os
import torch
import pickle
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def func1(input_path, output_path):

    # for name in ['train', 'dev', 'test.sighan15']:
    new_data = {}
    for key in ['id', 'src', 'tgt', 'tokens_size', 'src_idx', 'tgt_idx', 'lengths']:
        new_data[key] = []

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    for item in tqdm(data):
        for i in range(len(item['id'])):
            for key in new_data:
                if key == 'src_idx' or key == 'tgt_idx':
                    tmp = item[key][i].numpy().tolist()
                    value = []
                    for v in tmp:
                        if v > 0:
                            value.append(v)
                else:
                    value = item[key][i]  
                new_data[key].append(value)

    N = len(new_data['id'])
    for i in range(N):
        assert len(new_data['src_idx'][i]) == len(new_data['tgt_idx'][i])
        assert new_data['lengths'][i]+2 == len(new_data['tgt_idx'][i])
        assert new_data['lengths'][i] == len(new_data['tokens_size'][i]), new_data['tokens_size'][i]

    new_data_t = []
    for i in range(N):
        tmp_item = {}
        for key in new_data:
            tmp_item[key] = new_data[key][i]
        new_data_t.append(tmp_item)
    print(len(new_data_t))
    pickle.dump(new_data_t, open(output_path, 'wb'))

if __name__=='__main__':
    func1(sys.argv[1], sys.argv[2])


'''
python process_data.py /data/dobby_ceph_ir/hengdaxu/spell-acl-data/trainall.times1.32.pickle /data/dobby_ceph_ir/hengdaxu/spell-acl-data/trainall.times1.pkl
python process_data.py /data/dobby_ceph_ir/hengdaxu/spell-acl-data/trainall.times2.32.pickle /data/dobby_ceph_ir/hengdaxu/spell-acl-data/trainall.times2.pkl
python process_data.py /data/dobby_ceph_ir/hengdaxu/spell-acl-data/trainall.times3.32.pickle /data/dobby_ceph_ir/hengdaxu/spell-acl-data/trainall.times3.pkl
'''
