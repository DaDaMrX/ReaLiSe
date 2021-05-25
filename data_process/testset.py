'''Load & clean test data from SIGHAN14, SIGHAN15

Author: hengdaxu
 Email: hengdaxu@tencent.com

Cleaning processes:
    1. Make sure no spaces
    2. Traditional Chinese charactor to Simplified
        2.1 OpenCC converter
        2.2 著 -> 着, 妳 -> 你
    3. 「 -> “, 」 -> ”
    4. No English punctuation, only Chinese punctuation
    5. No symbol �
    6. End with Chinese punctuation
'''

import opencc

converter = opencc.OpenCC('t2s.json')


def full2half_width(text):

    def char_full2half_width(char):
        code = ord(char)
        if code == 0x3000:
            code = 0x20
        elif 0xff01 <= code <= 0xff5e:
            code -= 0xfee0
        return chr(code)

    res = []
    for c in text:
        if c.isalnum() or c in ['－', '．']:
            c = char_full2half_width(c)
        res.append(c)

    return ''.join(res)


def traditional_to_simple(text):
    text = converter.convert(text)
    text = text.replace('著', '着').replace('妳', '你')
    return text


def clean(text):
    text = text.replace('「', '“').replace('」', '”')
    text = text.replace('?', '？').replace(',', '，')
    text = full2half_width(text)
    return text


def find_words(s):
    def is_letter(c):
        return ord('a') <= ord(c.lower()) <= ord('z')
    l = 0
    while True:
        while l < len(s) and not is_letter(s[l]):
            l += 1
        if l == len(s):
            break
        r = l + 1
        while r < len(s) and is_letter(s[r]):
            r += 1
        yield l, r
        l = r


def load_test(input_path, label_path, year):
    assert year in [14, 15, 13]

    with open(input_path, 'r') as f:
        input_rows = [line.strip().split('\t') for line in f.read().splitlines()]
    with open(label_path, 'r') as f:
        label_rows = [line.strip().split(', ') for line in f.read().splitlines()]

    if year == 13:
        for row in input_rows:
            assert len(row) == 2
            sent = row[1]
            sent = sent.replace('…', '')
            sent = sent.replace('(', '').replace(')', '')
            row[1] = sent
    if year == 14:
        assert label_rows[491] == ['B1-1430-2', '8', '恤', '55', '恤']
        label_rows[491] = ['B1-1430-2', '0']

        assert label_rows[587] == ['B1-2164-1', '20', '爛']
        label_rows[587] = ['B1-2164-1', '0']

        assert input_rows[255][0] == '(pid=B1-0623-2)'
        assert input_rows[255][1].endswith('好。（剛剛考試的題目太難、太複雜了，讓我的心裡不太好）')
        input_rows[255][1] = input_rows[255][1].replace('（', '').replace('）', '')

        assert input_rows[491][0] == '(pid=B1-1430-2)'
        assert input_rows[491][1].endswith('開這麼貴的價錢．．．')
        input_rows[491][1] = input_rows[491][1].replace('．．．', '。')

        assert input_rows[957][0] == '(pid=B1-3917-2)'
        assert input_rows[957][1].endswith('是早一點睡！健康也好．．．')
        input_rows[957][1] = input_rows[491][1].replace('．．．', '。')
    if year == 15:
        assert input_rows[57][0] == '(pid=A2-0506-1)'
        assert input_rows[57][1] == '所以我在＂義大利麵方子＂已經定位了'
        input_rows[57][1] = '所以我在“義大利麵方子”已經定位了。'

        assert input_rows[600][0] == '(pid=B2-3625-3)'
        assert input_rows[600][1].endswith('家園，開開心心地看小孫子下課回來叫一聲＂爺爺＂。')
        input_rows[600][1] = input_rows[600][1].replace('一聲＂爺爺＂。', '一聲“爺爺”。')

        assert input_rows[998][0] == '(pid=B2-4252-7)'
        assert input_rows[998][1].startswith('如果你每天跟他們說＂你的父母')
        input_rows[998][1] = input_rows[998][1].replace('他們說＂你的父母', '他們說你的父母')

        assert input_rows[1088][0] == '(pid=B2-4393-2)'
        assert input_rows[1088][1].startswith('想一下就知道這樣不合理：＂對阿，我')
        input_rows[1088][1] = input_rows[1088][1].replace('理：＂對阿，我根', '理：“對阿，我根')
        input_rows[1088][1] = input_rows[1088][1].replace('相信我嗎？＂', '相信我嗎？”')

        assert input_rows[899][0] == '(pid=B2-4131-1)'
        assert input_rows[899][1].endswith('也很多（小孩）子，對他們真的不好吧！…')
        input_rows[899][1] = input_rows[899][1].replace('（', '').replace('）', '')
        input_rows[899][1] = input_rows[899][1].replace('的不好吧！…', '的不好吧！')

    assert len(input_rows) == len(label_rows)


    data = []
    for input_row, label_row in zip(input_rows, label_rows):
        assert len(input_row) == 2
        assert len(label_row) >= 2

        input_idx, src = input_row[0][5:-1], input_row[1]
        idx, tags = label_row[0], label_row[1:]
        assert idx == input_idx
        assert src.find(' ') == -1
        src = clean(src)

        mistakes = []
        if tags[0] == '0':
            assert len(tags) == 1
        else:
            for i in range(0, len(tags), 2):
                pos = int(tags[i]) - 1
                assert 0 <= pos < len(src)

                correct = tags[i + 1]
                assert len(correct) == 1
                if src[pos] == correct:
                    print('src[pos] == correct')
                    print(idx)
                    print(src)
                    print(tags)
                    print(tags[i], tags[i + 1])
                    print(src[pos], correct)
                    print()
                    continue
                assert src[pos] != correct

                mistakes.append({
                    'pos': pos,
                    'correct': correct,
                })

        item = {}
        data.append(item)

        # Field: id, src
        item['id'] = idx
        item['src'] = src

        # Field: tgt
        tgt = list(item['src'])
        for mis in mistakes:
            tgt[mis['pos']] = mis['correct']
        item['tgt'] = ''.join(tgt)

    for item in data:
        # Find words
        span_list, word_list = [], []
        for l, r in find_words(item['src']):
            span_list.append((l, r))
            if item['src'][l:r] not in word_list:
                word_list.append(item['src'][l:r])
        for i, (l, r) in enumerate(span_list):
            word = item['src'][l:r]
            idx = word_list.index(word)
            span_list[i] = (l, r, idx)
        src_list, tgt_list = list(item['src']), list(item['tgt'])
        for l, r, off in span_list:
            assert item['src'][l:r] == item['tgt'][l:r]
            src_list[l] = chr(9312 + off)
            tgt_list[l] = chr(9312 + off)
            for i in range(l + 1, r):
                src_list[i] = ''
                tgt_list[i] = ''
        item['src'] = ''.join(src_list)
        item['tgt'] = ''.join(tgt_list)

        # Remove space
        src_list, tgt_list = [], []
        for a, b in zip(item['src'], item['tgt']):
            if not a.isspace():
                assert not b.isspace()
                src_list.append(a)
                tgt_list.append(b)
            else:
                assert a == b
        item['src'] = ''.join(src_list)
        item['tgt'] = ''.join(tgt_list)

        # Make sure no English symbols
        for s in r'�．!@#$%^&*_+()=`~\|<>,/?:;\'"':
            assert item['src'].find(s) == -1
            assert item['tgt'].find(s) == -1

        # Make sure end with Chinese punctuation
        if not item['src'][-1] in r'。？！：”':
            item['src'] += '。'
            item['tgt'] += '。'

        # Traditional to simple
        item['src'] = traditional_to_simple(item['src'])
        item['tgt'] = traditional_to_simple(item['tgt'])

        # Field: errors
        errors = []
        for i, (a, b) in enumerate(zip(item['src'], item['tgt']), start=1):
            if a != b:
                errors.append((i, b))
        item['errors'] = str(errors)

    return data


def write_data(data, input_path, label_path):
    rows = ['\t'.join([item['id'], item['src'], item['tgt'],
                       item['errors']]) for item in data]
    with open(input_path, 'w') as f:
        f.write('\n'.join(rows))

    rows = []
    for item in data:
        row = [item['id']]
        if len(eval(item['errors'])) == 0:
            row.append('0')
        else:
            for i, c in eval(item['errors']):
                row.append(str(i))
                row.append(c)
        rows.append(', '.join(row))
    with open(label_path, 'w') as f:
        f.write('\n'.join(rows))


if __name__ == '__main__':
    # data = load_test(
    #     input_path='SIGHAN2015/Test/SIGHAN15_CSC_TestInput.txt',
    #     label_path='SIGHAN2015/Test/SIGHAN15_CSC_TestTruth.txt',
    #     year=15,
    # )

    # data = load_test(
    #     input_path='SIGHAN2013/FinalTest/SIGHAN13_TestInput.txt',
    #     label_path='SIGHAN2013/FinalTest/SIGHAN13_TestTruth.txt',
    #     year=14,
    # )

    data = load_test(
        input_path='SIGHAN2013/FinalTest/FinalTest_SubTask2.txt',
        label_path='SIGHAN2013/FinalTest/FinalTest_SubTask2_Truth.txt',
        year=13,
    )

    print('#sent:', len(data))
    print('n_sent_err:', sum(len(eval(s['errors'])) > 0 for s in data))
    print('n_err:', sum(len(eval(s['errors'])) for s in data))
    print('Ave len:', sum(len(s['src']) for s in data) / len(data))
    print('max len:', max(len(s['src']) for s in data))
    print('min len:', min(len(s['src']) for s in data))

    # write_data(
    #     data=data,
    #     input_path='data/test.sighan15.tsv',
    #     label_path='data/test.sighan15.lbl.tsv',
    # )

    # write_data(
    #     data=data,
    #     input_path='data/test.sighan14.tsv',
    #     label_path='data/test.sighan14.lbl.tsv',
    # )

    write_data(
        data=data,
        input_path='data/test.sighan13.tsv',
        label_path='data/test.sighan13.lbl.tsv',
    )
