import argparse
import os

from tqdm import tqdm
import numpy as np

from metric_core import read_file, sent_metric_detect, sent_metric_correct


def eval_measure(targ, pred):
    return sent_metric_detect(preds=pred, targs=targ)['sent-detect-f1']
    # return sent_metric_correct(preds=pred, targs=targ)['sent-correct-f1']


def sig_test_neubig(targ_path, pred1_path, pred2_path, num_samples, sample_ratio):
    gold = read_file(targ_path)
    sys1 = read_file(pred1_path)
    sys2 = read_file(pred2_path)
    assert len(gold) == len(sys1) == len(sys2)

    sys1_scores, sys2_scores = [], []
    wins = [0, 0, 0]
    ids = list(range(len(gold)))
    for _ in tqdm(range(num_samples)):
        np.random.shuffle(ids)
        reduced_ids = ids[:int(len(ids) * sample_ratio)]
        reduced_gold = [gold[i] for i in reduced_ids]
        reduced_sys1 = [sys1[i] for i in reduced_ids]
        reduced_sys2 = [sys2[i] for i in reduced_ids]
        sys1_score = eval_measure(reduced_gold, reduced_sys1)
        sys2_score = eval_measure(reduced_gold, reduced_sys2)
        if sys1_score > sys2_score:
            wins[0] += 1
        elif sys1_score < sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1
        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)

    wins = [x/float(num_samples) for x in wins]
    print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
        print('(sys1 is superior with p value p=%.3f)\n' % (1-wins[0]))
    elif wins[1] > wins[0]:
        print('(sys2 is superior with p value p=%.3f)\n' % (1-wins[1]))

    sys1_scores.sort()
    sys2_scores.sort()
    print('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
    print('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))


if __name__ == '__main__':
    paser = argparse.ArgumentParser()

    # paser.add_argument('--log_dir', default='logs5')
    # paser.add_argument('--t1', required=True)
    # paser.add_argument('--v1', type=int, default=0, required=True)
    # paser.add_argument('--c1', type=int, required=True)
    
    # paser.add_argument('--t2', required=True)
    # paser.add_argument('--v2', type=int, default=0, required=True)
    # paser.add_argument('--c2', type=int, required=True)

    paser.add_argument('--pred1_path', required=True)
    paser.add_argument('--pred2_path', required=True)
    paser.add_argument('--targ_path', default='our_truth/test.sighan15.lbl.tsv')
    paser.add_argument('--num_samples', type=int, default=10000)
    paser.add_argument('--sample_ratio', type=float, default=0.5)
    args = paser.parse_args()

    # pred1_path = os.path.join(
    #     args.log_dir,
    #     args.t1,
    #     f'version_{args.v1}',
    #     'results_sighan15',
    #     f'lbl_test_{args.c1}.txt',
    # )
    # pred2_path = os.path.join(
    #     args.log_dir,
    #     args.t2,
    #     f'version_{args.v2}',
    #     'results_sighan15',
    #     f'lbl_test_{args.c2}.txt',
    # )
    # print('pred1_path:', args.pred1_path)
    # print('pred2_path:', args.pred2_path)

    sig_test_neubig(
        targ_path=args.targ_path,
        pred1_path=args.pred1_path,
        pred2_path=args.pred2_path,
        num_samples=args.num_samples,
        sample_ratio=args.sample_ratio,
    )
