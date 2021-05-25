import torch


def add_trans_to_weights(input_weights_path, bert_weights_path, output_weights_path):
    weights = torch.load(input_weights_path)
    bert_weights = torch.load(bert_weights_path)
    trans_state_dict = {'bert_trans.' + '.'.join(k.split('.')[3:]): v for k, v in bert_weights.items() if k.split('.')[:3] == ['cls', 'predictions', 'transform']}
    weights.update(trans_state_dict)
    torch.save(weights, output_weights_path)


if __name__ == '__main__':
    add_trans_to_weights(
        input_weights_path='/data/dobby_ceph_ir/neutrali/pretrained_models/zh-spelling-corr/pretrained-pho2res_pho-bs-64-mxs-30000_res-i0-v1/pytorch_model.bin',
        bert_weights_path='/data/dobby_ceph_ir/hengdaxu/.local/chinese_wwm_ext/pytorch_model.bin',
        output_weights_path='/data/dobby_ceph_ir/hengdaxu/pretrained_models/pretrained-pho2res_pho-bs-64-mxs-30000_res-i0-v1-trans/pytorch_model.bin',
    )
