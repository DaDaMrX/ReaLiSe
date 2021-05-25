def build_lbl(data_path, lbl_path):
    with open(data_path) as f:
        rows = [s.split('\t') for s in f.read().splitlines()]

    data = []
    for idx, src, tgt, errors in rows:
        item = [idx]
        errors = eval(errors)
        if len(errors) > 0:
            for pos, correct in errors:
                item.append(str(pos))
                item.append(correct)
        else:
            item.append('0')
        data.append(', '.join(item))

    with open(lbl_path, 'w') as f:
        f.write('\n'.join(data))
        


if __name__ == '__main__':
    build_lbl(
        data_path='data/dev.tsv',
        lbl_path='data/dev.lbl.tsv',
    )
