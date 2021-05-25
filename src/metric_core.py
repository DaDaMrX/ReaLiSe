import argparse


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = [r.strip().split(', ') for r in f.read().splitlines()]

    data = []
    for row in rows:
        item = [row[0]]
        data.append(item)
        if len(row) == 2 and row[1] == '0':
            continue
        for i in range(1, len(row), 2):
            item.append((int(row[i]), row[i + 1]))

    return data


def metric_file(pred_path, targ_path, do_char_metric):
    preds = read_file(pred_path)
    targs = read_file(targ_path)

    results = {}
    res = sent_metric_detect(preds=preds, targs=targs)
    results.update(res)
    res = sent_metric_correct(preds=preds, targs=targs)
    results.update(res)

    if do_char_metric:
        res = char_metric(preds=preds, targs=targs)
        results.update(res)

    return results


def sent_metric_detect(preds, targs):
    assert len(preds) == len(targs)
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for pred_item, targ_item in zip(preds, targs):
        assert pred_item[0] == targ_item[0]
        pred, targ = sorted(pred_item[1:]), sorted(targ_item[1:])
        if targ != []:
            targ_p += 1
        if pred != []:
            pred_p += 1
        if len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)):
            hit += 1
        if pred != [] and len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)):
            tp += 1

    acc = hit / len(targs)
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = {
        'sent-detect-acc': acc * 100,
        'sent-detect-p': p * 100,
        'sent-detect-r': r * 100,
        'sent-detect-f1': f1 * 100,
    }
    return results


def sent_metric_correct(preds, targs):
    assert len(preds) == len(targs)
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for pred_item, targ_item in zip(preds, targs):
        assert pred_item[0] == targ_item[0]
        pred, targ = sorted(pred_item[1:]), sorted(targ_item[1:])
        if targ != []:
            targ_p += 1
        if pred != []:
            pred_p += 1
        if pred == targ:
            hit += 1
        if pred != [] and pred == targ:
            tp += 1

    acc = hit / len(targs)
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = {
        'sent-correct-acc': acc * 100,
        'sent-correct-p': p * 100,
        'sent-correct-r': r * 100,
        'sent-correct-f1': f1 * 100,
    }
    return results


def char_detect(pred, targ):
    pred = [p for p, c in pred]
    targ = [p for p, c in targ]
    ps, ts = [], []
    for x in set(pred + targ):
        ps.append(x in pred)
        ts.append(x in targ)

    tp, fp, tn, fn = 0, 0, 0, 0
    for p, t in zip(ps, ts):
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif not p and not t:
            tn += 1
        else:
            fn += 1

    assert tn == 0
    return tp, fp, tn, fn


def char_correct(pred, targ):
    pred_d = {p: c for p, c in pred}
    targ_d = {p: c for p, c in targ}
    pred = [p for p, c in pred]
    targ = [p for p, c in targ]

    tp, fp, tn, fn = 0, 0, 0, 0
    for x in set(pred + targ):
        if x in targ and x in pred and targ_d[x] == pred_d[x]:
            tp += 1
        elif x in targ and x in pred and targ_d[x] != pred_d[x]:
            fp += 1  # predict error (affect precision)
            fn += 1  # didn't recall (affect recall)
        elif x in targ and x not in pred:
            fn += 1
        elif x not in targ and x in pred:
            fp += 1
        else:
            tn += 1

    assert tn == 0
    return tp, fp, tn, fn


def char_metric(preds, targs):
    assert len(preds) == len(targs)
    char_results = {}

    # Detect
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred_item, targ_item in zip(preds, targs):
        assert pred_item[0] == targ_item[0]
        pred, targ = sorted(pred_item[1:]), sorted(targ_item[1:])
        tp_, fp_, tn_, fn_ = char_detect(pred=pred, targ=targ)
        tp, fp, tn, fn = tp + tp_, fp + fp_, tn + tn_, fn + fn_

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = {
        'char-detect-p': p * 100,
        'char-detect-r': r * 100,
        'char-detect-f1': f1 * 100,
    }
    char_results.update(results)

    # Correct
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred_item, targ_item in zip(preds, targs):
        assert pred_item[0] == targ_item[0]
        pred, targ = sorted(pred_item[1:]), sorted(targ_item[1:])
        tp_, fp_, tn_, fn_ = char_correct(pred=pred, targ=targ)
        tp, fp, tn, fn = tp + tp_, fp + fp_, tn + tn_, fn + fn_

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = {
        'char-correct-p': p * 100,
        'char-correct-r': r * 100,
        'char-correct-f1': f1 * 100,
    }
    char_results.update(results)

    return char_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--target', '-t', required=True)
    parser.add_argument('--do_char_metric', action='store_true')
    args = parser.parse_args()

    results = metric_file(
        pred_path=args.input,
        targ_path=args.target,
        do_char_metric=args.do_char_metric,
    )

    # assert results['sent-detect-acc'] == 0.6
    # assert results['sent-detect-p'] == 0.8
    # assert results['sent-detect-r'] == 0.5714285714285714
    # assert results['sent-detect-f1'] == 0.6666666666666666
    # assert results['sent-correct-f1'] == 0.5454545454545454
    # assert results['char-detect-f1'] == 0.761904761904762
    # assert results['char-correct-f1'] == 0.7000000000000001
    # print('Test Pass!')

    for k, v in results.items():
        print(f'{k}: {v}')

'''
python src/metric_core.py \
    -t tmp_metric_data/test.sighan15.lbl.tsv \
    -i tmp_metric_data/labels.txt \
    --do_char_metric
'''
