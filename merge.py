import os
import torch


def merge(pho_model_path, res_model_path, output_path, sec_version=0):
    bert_state_dict = torch.load('/data/dobby_ceph_ir/neutrali/pretrained_models/roberta-base-ch-for-csc/pytorch_model.bin')
    pho_state_dict = torch.load(pho_model_path)
    res_state_dict = torch.load(res_model_path)

    if sec_version == 1:
        cur_res_keys = [key for key in res_state_dict.keys()]
        for key in cur_res_keys:
            if key.startswith('resnet.'):
                new_key = key.replace('resnet.', 'char_resent.')
                res_state_dict[new_key] = res_state_dict.pop(key)


    for key in pho_state_dict.keys():
        pho_state_dict[key] = pho_state_dict[key].to('cpu')
        bert_state_dict[key] = pho_state_dict[key]

    for key in res_state_dict.keys():
        res_state_dict[key] = res_state_dict[key].to('cpu')
        bert_state_dict[key] = res_state_dict[key]

    remove_keys = []
    for key in bert_state_dict.keys():
        if key.startswith('position_embeddings.') or key.startswith('char_images.'):
            remove_keys.append(key)    

    for key in remove_keys:
        print('Deleting', key)
        del bert_state_dict[key]

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(bert_state_dict, output_path)

if __name__=='__main__':
    pho_model = '/data/dobby_ceph_ir/neutrali/venus_outputs/spell-pretrain_model-pho2-pretrain_bs-%d_lr-5e-5_mxs-%d000_seed-42/pytorch_model.bin'
    res_model = '/data/dobby_ceph_ir/hengdaxu/venus_outputs/pretrain_res/pretrain_res_seed42_%s/pytorch_model.bin'
    out_path = '/data/dobby_ceph_ir/hengdaxu/venus_outputs/pretrain/pretrain_wwm_phob-%d_phos-%d_res-%s/pytorch_model.bin'
    for bs in [64]:
        for mxs in [15, 20, 30]:
            for resv in ['epoch8_font2', 'epoch8_font2_fanti', 'epoch8_font3_fanti', 'epoch15_font2', 'epoch15_font2_fanti', 'epoch15_font3_fanti']:
                print('Merge:', pho_model%(bs, mxs), res_model%resv, '...')
                merge(pho_model%(bs, mxs), res_model%resv, out_path%(bs, mxs, resv))

    # Ablation
    # pho_model = '/data/dobby_ceph_ir/neutrali/venus_outputs/spell-pretrain_model-pho2-pretrain_bs-%d_lr-5e-5_mxs-%d000_seed-42/pytorch_model.bin'
    # res_model = '/data/dobby_ceph_ir/hengdaxu/venus_outputs/pretrain_res/pretrain_res_seed42_%s/pytorch_model.bin'
    # out_path = '/data/dobby_ceph_ir/hengdaxu/venus_outputs/pretrain/pretrain_wwm_phob-%d_phos-%d_res-%s/pytorch_model.bin'
    # for bs in [64]:
    #     for mxs in [30]:
    #         for resv in ['epoch8_font1']:
    #             print('Merge:', pho_model%(bs, mxs), res_model%resv, '...')
    #             merge(pho_model%(bs, mxs), res_model%resv, out_path%(bs, mxs, resv))
