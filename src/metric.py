import os
import transformers

from metric_core import metric_file
from remove_de import remove_de


class Metric:

    def __init__(self, vocab_path):
        self.tokenizer = transformers.BertTokenizer.from_pretrained(vocab_path)

    def metric(self, batches, pred_txt_path, pred_lbl_path, label_path, should_remove_de=False):
        self.write_pred(batches, pred_txt_path, pred_lbl_path)
        if should_remove_de:
            remove_de(
                input_path=pred_lbl_path,
                output_path=pred_lbl_path,
            )
        scores = metric_file(
            pred_path=pred_lbl_path,
            targ_path=label_path,
            do_char_metric=False,
        )
        return scores

    def write_pred(self, batches, pred_txt_path, pred_lbl_path):
        pred_txt_list, pred_lbl_list = [], []
        for batch in batches:
            for i in range(batch['src_idx'].shape[0]):
                pred_txt, pred_lbl = self.process_batch_item(batch, i)
                pred_txt_list.append(pred_txt)
                pred_lbl_list.append(pred_lbl)

        pred_dir = os.path.dirname(pred_lbl_path)
        os.makedirs(pred_dir, exist_ok=True)

        with open(pred_lbl_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(pred_lbl_list))
        print('\n\n')
        print(f'Metric write to "{pred_lbl_path}"')

        with open(pred_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(pred_txt_list))
        print(f'Metric write to "{pred_txt_path}"')

    def process_batch_item(self, batch, idx):
        length = batch['lengths'][idx]

        pred_idx = batch['pred_idx'][idx].tolist()
        pred_idx = pred_idx[1:1 + length]
        tokens = self.tokenizer.convert_ids_to_tokens(pred_idx)
        tokens = [t if not t.startswith('##') else t[2:] for t in tokens]
        tokens = [t if t != self.tokenizer.unk_token else 'U' for t in tokens]
        t_tokens = []
        for s, t in zip(batch['tokens_size'][idx], tokens):
            token = t[:s]
            if len(token) < s:
                token += 'x' * (s - len(token))
            t_tokens.append(token)
        pred = ''.join(t_tokens)
        pred_txt = batch['id'][idx] + '\t' + pred

        src = batch['src'][idx]
        if len(src) > len(pred):
            src = src[:len(pred)]
        assert len(pred) == len(src)
        
        item = [batch['id'][idx]]
        for i, (a, b) in enumerate(zip(src, pred), start=1):
            if a != b:
                item.append(str(i))
                item.append(b)
        if len(item) == 1:
            item.append('0')
        pred_lbl = ', '.join(item)

        return pred_txt, pred_lbl
