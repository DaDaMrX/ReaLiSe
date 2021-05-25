import torch
import pypinyin


class Pinyin(object):
    """docstring for Pinyin"""
    def __init__(self):
        super(Pinyin, self).__init__()
        self.shengmu = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
        self.yunmu = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iu', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'ue', 'ui', 'un', 'uo', 'v', 've']
        self.pho_vocab_list = ['[PAD]', '[NULL]', '[UNK]']
        self.pho_vocab_list += self.shengmu + self.yunmu
        self.pho_vocab_list += ['1', '2', '3', '4', '5']
        self.pho_vocab = {}
        for i, p in enumerate(self.pho_vocab_list):
            self.pho_vocab[p] = i

    def get_pho_size(self):
        return len(self.pho_vocab_list)

    def get_pinyin(self, c):
        if len(c) > 1:
            return '[UNK]', '[UNK]', '[UNK]'
        if c == '嗯':
            return '[NULL]', 'en', 2
        s = pypinyin.pinyin(
            c,
            style=pypinyin.Style.TONE3,
            neutral_tone_with_five=True,
            errors=lambda x: ['U' for _ in x],
        )[0][0]
        if s == 'U':
            return '[UNK]', '[UNK]', '[UNK]'
        assert isinstance(s, str)
        assert s[-1] in '12345'
        sm, ym, sd = '[NULL]', None, None
        for c in self.shengmu:
            if s.startswith(c):
                sm = c
                break
        if sm == '[NULL]':
            ym = s[:-1]
        else:
            ym = s[len(sm):-1]
        sd = s[-1]
        return sm, ym, sd

    def convert(self, tokens):
        results = []
        unk_id = self.pho_vocab.get('[UNK]')
        for token in tokens:
            sm, ym, sd = self.get_pinyin(token)
            results.append((self.pho_vocab.get(sm, unk_id), self.pho_vocab.get(ym, unk_id), self.pho_vocab.get(sd, unk_id)))
        return results
pho_convertor = Pinyin()


class Pinyin2(object):
    def __init__(self):
        super(Pinyin2, self).__init__()
        pho_vocab = ['P']
        pho_vocab += [chr(x) for x in range(ord('1'), ord('5') + 1)]
        pho_vocab += [chr(x) for x in range(ord('a'), ord('z') + 1)]
        pho_vocab += ['U']
        assert len(pho_vocab) == 33
        self.pho_vocab_size = len(pho_vocab)
        self.pho_vocab = {c: idx for idx, c in enumerate(pho_vocab)}

    def get_pho_size(self):
        return self.pho_vocab_size

    @staticmethod
    def get_pinyin(c):
        if len(c) > 1:
            return 'U'
        s = pypinyin.pinyin(
            c,
            style=pypinyin.Style.TONE3,
            neutral_tone_with_five=True,
            errors=lambda x: ['U' for _ in x],
        )[0][0]
        if s == 'U':
            return s
        assert isinstance(s, str)
        assert s[-1] in '12345'
        s = s[-1] + s[:-1]
        return s

    def convert(self, chars):
        pinyins = list(map(self.get_pinyin, chars))
        pinyin_ids = [list(map(self.pho_vocab.get, pinyin)) for pinyin in pinyins]
        pinyin_lens = [len(pinyin) for pinyin in pinyins]
        pinyin_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in pinyin_ids],
            batch_first=True,
            padding_value=0,
        )
        return pinyin_ids, pinyin_lens
pho2_convertor = Pinyin2()

if __name__=='__main__':
    print(pho_convertor.get_pho_size())
    results = pho_convertor.convert([c for c in '测试一下'])
    for v1, v2, v3 in results:
        print(pho_convertor.pho_vocab_list[v1], pho_convertor.pho_vocab_list[v2], pho_convertor.pho_vocab_list[v3])