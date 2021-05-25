'''Load & clean train data from SIGHAN13, SIGHAN14, SIGHAN15 & Wang2019

Author: hengdaxu
 Email: hengdaxu@tencent.com

Cleaning processes:
    1. Remove spaces
    2. Traditional Chinese charactor to Simplified
        2.1 OpenCC converter
        2.2 著 -> 着, 妳 -> 你
    3. 「 -> “, 」 -> ”
    4. No English punctuation, only Chinese punctuation
    5. No symbol �
    6. End with Chinese punctuation
'''

import re
import xml.etree.ElementTree as ET

import opencc
from tqdm import tqdm

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
    tra_text = text
    text = converter.convert(tra_text)
    text = text.replace('著', '着').replace('妳', '你')
    if '𪲔' in text:
        text = ''.joi([b if b != '𪲔' else a for a, b in zip(tra_text, text)])
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


def fix_data_train_13(text):
    text = text.replace(
        '對我洗腦，我�堅定的心，就這樣被他所動遙了。</P>',
        '對我洗腦，我堅定的心，就這樣被他所動遙了。</P>',
    )
    text = text.replace(
        '<MISTAKE wrong_position=64>\n'
        '<WRONG>動遙</WRONG>\n'
        '<CORRECT>動搖</CORRECT>\n',
        '<MISTAKE wrong_position=63>\n'
        '<WRONG>動遙</WRONG>\n'
        '<CORRECT>動搖</CORRECT>\n',
    )
    text = text.replace(
        '<MISTAKE wrong_position=16>\n'
        '<WRONG>輕意</WRONG>\n'
        '<CORRECT>輕易</CORRECT>\n',
        '<MISTAKE wrong_position=17>\n'
        '<WRONG>輕意</WRONG>\n'
        '<CORRECT>輕易</CORRECT>\n',
    )
    text = text.replace(
        '<MISTAKE wrong_position=21>\n'
        '<WRONG>徬惶</WRONG>\n'
        '<CORRECT>徬徨</CORRECT>\n',
        '<MISTAKE wrong_position=22>\n'
        '<WRONG>徬惶</WRONG>\n'
        '<CORRECT>徬徨</CORRECT>\n',
    )
    return text


def load_train_13(path):
    with open(path, 'rb') as f:
        text = f.read().decode(errors='replace')
    text = '<xml>' + text + '</xml>'

    # Fix data
    if 'WithError' in path:
        text = fix_data_train_13(text)
    text = re.sub(r'wrong_position=(.*)>', r'wrong_position="\1">', text)

    root = ET.fromstring(text)

    data = []
    for doc in root:
        item = {}
        data.append(item)

        # Field: id
        item['id'] = doc.get('Nid').strip()
        assert item['id'].isnumeric() and len(item['id']) == 5
        item['id'] = 'sighan13-' + item['id']

        # Field: src
        src = doc.find('P').text.strip()
        src = src.replace(' ', '')
        src = clean(src)
        assert len(src) >= 2
        item['src'] = src

        # mistakes
        mistakes = []
        for mistake in doc.find('TEXT'):
            pos = int(mistake.get('wrong_position')) - 1
            if pos == -1:
                continue
            assert 0 <= pos < len(src)

            wrong = mistake.find('WRONG').text.strip()
            wrong = clean(wrong)
            correct = mistake.find('CORRECT').text.strip()
            correct = clean(correct)
            assert src.find(wrong) > -1
            assert len(wrong) == len(correct) > 0

            left_pos = item['src'].find(wrong)
            right_pos = left_pos + len(wrong) - 1
            assert left_pos <= pos <= right_pos
            assert correct[pos - left_pos] != item['src'][pos]

            mistakes.append({
                'pos': pos,
                'wrong': wrong,
                'correct': correct,
                'left_pos': left_pos,
                'right_pos': right_pos,
            })

        # Field: tgt
        tgt = list(item['src'])
        for mis in mistakes:
            for i, w, c in zip(
                range(mis['left_pos'], mis['right_pos'] + 1),
                mis['wrong'],
                mis['correct'],
            ):
                assert tgt[i] == w
                tgt[i] = c
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

        # Whitespace
        assert not any(c.isspace() for c in item['src'])
        assert not any(c.isspace() for c in item['tgt'])

        # Make sure no English symbols
        for s in r'�．!@#$%^&*()_+=`~\|<>,/?:;\'"':
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


def fix_data_train_14_B1(text):
    # Replace 12 �
    text = text.replace(
        '他們多很高興�以我也陪他們高空彈跳。</PASSAGE>',
        '他們多很高興所以我也陪他們高空彈跳。</PASSAGE>',
    )
    text = text.replace(
        '<WRONG>根也是一個能賺錢��方法</WRONG>',
        '<WRONG>根也是一個能賺錢的方法</WRONG>'
    )
    text = text.replace(
        '因為哪裡什麼花都沒有，所以有�點兒奇怪，可是我更喜歡看樹',
        '因為哪裡什麼花都沒有，所以有一點兒奇怪，可是我更喜歡看樹',
    )
    text = text.replace(
        '<WRONG>我�的班的同學在台灣學中文含我有些同學</WRONG>',
        '<WRONG>我們的班的同學在台灣學中文含我有些同學</WRONG>'
    )
    text = text.replace(
        '<PASSAGE id="B1-1388-1">我在網路上買了新的電子辭典，因為�網路上買的話',
        '<PASSAGE id="B1-1388-1">我在網路上買了新的電子辭典，因為在網路上買的話',
    )
    text = text.replace(
        '我去過森林�市、淡水',
        '我去過森林都市、淡水',
    )
    text = text.replace(
        '<PASSAGE id="B1-2358-1">因為我家�近有大安公園',
        '<PASSAGE id="B1-2358-1">因為我家附近有大安公園',
    )
    text = text.replace(
        '<PASSAGE id="B1-3102-2">因為我知道他們�戀愛',
        '<PASSAGE id="B1-3102-2">因為我知道他們的戀愛',
    )
    text = text.replace(
        '還有��多好朋友們等等。</PASSAGE>',
        '還有很多好朋友們等等。</PASSAGE>',
    )
    text = text.replace(
        '著，�自己要有信心不要為了小事而害上我們的身體。</PASSAGE>',
        '著，对自己要有信心不要為了小事而害上我們的身體。</PASSAGE>',
    )
    assert text.find('�') == -1

    text = text.replace(
        '<CORRECTION>跟也是一個能賺錢的方法</CORRECTION>',
        '<CORRECTION>这也是一個能賺錢的方法</CORRECTION>',
    )
    text = text.replace(
        '<CORRECTION>累地我把門打開</CORRECTION>',
        '<CORRECTION>累得我把門打開</CORRECTION>',
    )
    text = text.replace(
        '<MISTAKE id="B1-3202-1" location="19">',
        '<MISTAKE id="B1-3202-1" location="35">'
    )
    text = text.replace(
        '<MISTAKE id="B1-2119-2" location="38">',
        '<MISTAKE id="B1-2119-2" location="11">'
    )
    # Length
    text = text.replace(
        '<CORRECTION>挑戰性心</CORRECTION>',
        '<CORRECTION>挑戰性</CORRECTION>',
    )
    text = text.replace(
        '<CORRECTION>過時間</CORRECTION>',
        '<CORRECTION>過的時間</CORRECTION>',
    )
    # Punctuation
    text = text.replace(
        '真的是人山人海.我不知道我在哪裡。</PASSAGE>',
        '真的是人山人海，我不知道我在哪裡。</PASSAGE>',
    )
    text = text.replace(
        '也幫我替你爸媽好!！</PASSAGE>',
        '也幫我替你爸媽好！</PASSAGE>',
    )
    text = text.replace(
        '前三部！但衣服店是滿多了。]</PASSAGE>',
        '前三部！但衣服店是滿多了。</PASSAGE>'
    )
    text = text.replace(
        '大學，見到他我非常高興，</PASSAGE>',
        '大學，見到他我非常高興。</PASSAGE>'
    )
    # Repeated char in wrong
    text = text.replace(
        '<MISTAKE id="B1-1607-3" location="11">',
        '<MISTAKE id="B1-1607-3" location="12">',
    )
    text = text.replace(
        '<MISTAKE id="B1-2399-3" location="9">',
        '<MISTAKE id="B1-2399-3" location="11">',
    )
    text = text.replace(
        '<MISTAKE id="B1-2598-2" location="16">',
        '<MISTAKE id="B1-2598-2" location="18">',
    )
    return text


def fix_data_train_14_C1(text):
    text = text.replace(
        '<MISTAKE id="C1-1800-2" location="29">',
        '<MISTAKE id="C1-1800-2" location="22">',
    )
    return text


def fix_data_train_15_A2(text):
    text = text.replace(
        '<ESSAY title="難忘的旅遊經驗">\n'
        '<TEXT>\n'
        '<PASSAGE id="A2-0782-1">走路的時候他試試看廳路上的汽車，'
        '就一位先生廳還告訴對我弟弟，他也到英國去，所以我弟弟可以跟他一起走。</PASSAGE>\n'
        '</TEXT>\n'
        '<MISTAKE id="A2-0782-1" location="10">\n'
        '<WRONG>廳路上</WRONG>\n'
        '<CORRECTION>聽路上</CORRECTION>\n'
        '</MISTAKE>\n'
        '<MISTAKE id="A2-0782-1" location="22">\n'
        '<WRONG>廰</WRONG>\n'
        '<CORRECTION>停</CORRECTION>\n'
        '</MISTAKE>\n'
        '</ESSAY>\n',
        ''
    )
    text = text.replace(
        '<MISTAKE id="A2-1291-1" location="16">',
        '<MISTAKE id="A2-1291-1" location="15">',
    )
    text = text.replace(
        '<MISTAKE id="A2-3313-1" location="14">',
        '<MISTAKE id="A2-3313-1" location="1">',
    )
    text = text.replace(
        '<PASSAGE id="A2-0087-3">她提以他們五點晚上去電影院看一個新電影．</PASSAGE>',
        '<PASSAGE id="A2-0087-3">她提以他們五點晚上去電影院看一個新電影。</PASSAGE>',
    )
    text = text.replace(
        '<MISTAKE id="A2-3380-1" location="13">',
        '<MISTAKE id="A2-3380-1" location="14">',
    )
    return text


def fix_data_train_15_B2(text):
    text = text.replace(
        '<PASSAGE id="B2-1454-6">此至，祝大安</PASSAGE>',
        '<PASSAGE id="B2-1454-5">此至，祝大安。</PASSAGE>',
    )
    text = text.replace(
        '<PASSAGE id="B2-3859-6">我覺得在網路上很',
        '<PASSAGE id="B2-3859-5">我覺得在網路上很',
    )
    text = text.replace(
        '<PASSAGE id="B2-4303-3">當然老',
        '<PASSAGE id="B2-4303-2">當然老',
    )
    text = text.replace(
        '<CORRECTION>同樣</CORRECTION>',
        '<CORRECTION>同樣地</CORRECTION>',
    )
    text = text.replace(
        '<WRONG>須機</WRONG>',
        '<WRONG>須要</WRONG>',
    )
    text = text.replace(
        '<MISTAKE id="B2-1683-2" location="1">',
        '<MISTAKE id="B2-1683-2" location="7">',
    )
    text = text.replace(
        '<MISTAKE id="B2-1683-4" location="31">',
        '<MISTAKE id="B2-1683-4" location="35">',
    )
    text = text.replace(
        '<MISTAKE id="B2-1978-4" location="24">\n'
        '<WRONG>華連</WRONG>\n'
        '<CORRECTION>花蓮</CORRECTION>\n'
        '</MISTAKE>\n',
        '',
    )
    text = text.replace(
        '<MISTAKE id="B2-2427-1" location="21">\n'
        '<WRONG>天天餵牠吃</WRONG>\n'
        '<CORRECTION> </CORRECTION>\n'
        '</MISTAKE>\n',
        '<MISTAKE id="B2-2427-1" location="33">\n'
        '<WRONG>天天為牠吃</WRONG>\n'
        '<CORRECTION>天天餵牠吃</CORRECTION>\n'
        '</MISTAKE>\n',
    )
    text = text.replace(
        '<MISTAKE id="B2-3666-4" location="10">\n'
        '<WRONG>他有沒有</WRONG>\n'
        '<CORRECTION>她有沒有</CORRECTION>\n'
        '</MISTAKE>\n'
        '<MISTAKE id="B2-3666-4" location="24">\n'
        '<WRONG>他不需要上班</WRONG>\n'
        '<CORRECTION>她不需要上班</CORRECTION>\n'
        '</MISTAKE>\n',
        '',
    )
    text = text.replace(
        '<MISTAKE id="B2-3666-4" location="24">\n'
        '<WRONG>做他愛做的事情</WRONG>\n'
        '<CORRECTION>做她愛做的事情</CORRECTION>\n'
        '</MISTAKE>\n',
        '',
    )
    text = text.replace(
        '<MISTAKE id="B2-3772-1" location="22">',
        '<MISTAKE id="B2-3772-1" location="15">',
    )
    text = text.replace(
        '<MISTAKE id="B2-3772-2" location="16">',
        '<MISTAKE id="B2-3772-2" location="22">',
    )
    text = text.replace(
        '<MISTAKE id="B2-3772-4" location="13">',
        '<MISTAKE id="B2-3772-4" location="16">',
    )
    text = text.replace(
        '<WRONG>圍週</WRONG>\n'
        '<CORRECTION>圍周</CORRECTION>\n',
        '<WRONG>圍周</WRONG>\n'
        '<CORRECTION>圍週</CORRECTION>\n',
    )
    text = text.replace(
        '<PASSAGE id="B2-4022-3">我們提針下列方法、加一張壁板在',
        '<PASSAGE id="B2-4022-3">我們提針下列方法：加一張壁板在',
    )
    text = text.replace(
        '<MISTAKE id="B2-4028-3" location="32">',
        '<MISTAKE id="B2-4028-3" location="30">',
    )
    text = text.replace(
        '把自己跟被偷東西的人換位子想。心</PASSAGE>',
        '把自己跟被偷東西的人換位子想。</PASSAGE>',
    )
    text = text.replace(
        '方說空氣阿、水阿、土地阿、越來越壞掉了。]</PASSAGE>',
        '方說空氣阿、水阿、土地阿、越來越壞掉了。</PASSAGE>',
    )
    text = text.replace(
        '前的那麼好。他真的賠了夫人又折兵﹗</PASSAGE>',
        '前的那麼好。他真的賠了夫人又折兵！</PASSAGE>',
    )
    text = text.replace(
        '<MISTAKE id="B2-4327-3" location="26">',
        '<MISTAKE id="B2-4327-3" location="30">',
    )
    text = text.replace(
        '<PASSAGE id="B2-4350-2">我想網站也��一個東西很好的，',
        '<PASSAGE id="B2-4350-2">我想網站也是一個東西很好的，',
    )
    return text


def load_train_14_15(path, year):
    assert year in [14, 15]
    with open(path, 'rb') as f:
        text = f.read().decode(errors='replace')
    text = '<xml>' + text + '</xml>'

    # Fix data
    if year == 14 and 'B1' in path:
        text = fix_data_train_14_B1(text)
    if year == 14 and 'C1' in path:
        text = fix_data_train_14_C1(text)
    if year == 15 and 'A2' in path:
        text = fix_data_train_15_A2(text)
    if year == 15 and 'B2' in path:
        text = fix_data_train_15_B2(text)

    root = ET.fromstring(text)

    data = []
    for essay in root.findall('ESSAY'):
        sents_dict = {}
        for passage in essay.find('TEXT').findall('PASSAGE'):
            # idx
            idx = passage.get('id').strip()
            assert len(idx) in [9, 10]

            # src
            src = passage.text.strip()
            src = clean(src)
            assert len(src) >= 2

            sents_dict[idx] = {'src': src, 'mistakes': []}

        for mistake in essay.findall('MISTAKE'):
            idx = mistake.get('id').strip()
            assert len(idx) in [9, 10] and idx in sents_dict
            src = sents_dict[idx]['src']

            pos = int(mistake.get('location')) - 1
            assert 0 <= pos < len(src)

            wrong = mistake.find('WRONG').text.strip()
            wrong = clean(wrong)
            correct = mistake.find('CORRECTION').text.strip()
            correct = clean(correct)

            assert src.find(wrong) > -1
            assert len(wrong) == len(correct) > 0

            # Find left_pos & right_pos
            if src.count(wrong) == 1:
                left_pos = src.find(wrong)
                right_pos = left_pos + len(wrong) - 1
            else:
                start_pos = 0
                while True:
                    left_pos = src.find(wrong, start_pos)
                    assert left_pos >= 0
                    right_pos = left_pos + len(wrong) - 1
                    if left_pos <= pos <= right_pos:
                        break
                    start_pos = left_pos + 1

            assert left_pos >= 0
            assert src[left_pos:right_pos + 1] == wrong
            assert left_pos <= pos <= right_pos
            # Cannot guarantee for sighan15-2
            if not (year == 15 and 'B2' in path):
                src[pos] != correct[pos - left_pos]

            sents_dict[idx]['mistakes'].append({
                'pos': pos,
                'wrong': wrong,
                'correct': correct,
                'left_pos': left_pos,
                'right_pos': right_pos,
            })

        for idx, sent_dict in sents_dict.items():
            item = {}
            data.append(item)

            # Field: id, src
            item['id'] = f'sighan{year}-{idx}'
            item['src'] = sent_dict['src']

            # Field: tgt
            tgt = list(item['src'])
            for mis in sent_dict['mistakes']:
                for i, w, c in zip(
                    range(mis['left_pos'], mis['right_pos'] + 1),
                    mis['wrong'],
                    mis['correct'],
                ):
                    if not (tgt[i] == w or tgt[i] == c):
                        print(item['id'], item['src'])
                        print(i, tgt[i], w, c)
                    assert tgt[i] == w or tgt[i] == c
                    tgt[i] = c
            item['tgt'] = ''.join(tgt)

    # Clean
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
            if a.isspace():
                assert b.isspace()
                continue
            src_list.append(a)
            tgt_list.append(b)
        item['src'] = ''.join(src_list)
        item['tgt'] = ''.join(tgt_list)

        # Make sure no English symbols
        for s in r'�．!@#$%^&*()_+=`~\|<>,/?:;\'"':
            assert item['src'].find(s) == -1
            assert item['tgt'].find(s) == -1

        # Make sure end with Chinese punctuation
        if not item['src'][-1] in r'.。？！：”':
            # print(item['src'])
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


def load_train_wang27k(path):
    with open(path, 'r') as f:
        text = f.read()
    text = '<xml>' + text + '</xml>'
    root = ET.fromstring(text)

    data = []
    for idx, doc in enumerate(tqdm(root)):
        item = {}
        data.append(item)

        # Field: id
        item['id'] = f'wang27k-{idx:06}'

        # Field: src
        src = doc.find('TEXT').text.strip()
        src = clean(src)
        assert len(src) >= 2
        assert src.find(' ') == -1
        item['src'] = src

        # mistakes
        mistakes = []
        for mistake in doc.findall('MISTAKE'):
            wrong = mistake.find('WRONG').text.strip()
            assert len(wrong) == 1
            assert item['src'].find(wrong) > -1

            correct = mistake.find('CORRECTION').text.strip()
            assert len(correct) == 1

            pos = int(mistake.find('LOCATION').text) - 1
            assert 0 <= pos < len(item['src'])
            assert item['src'][pos] == wrong != correct

            mistakes.append({
                'pos': pos,
                'wrong': wrong,
                'correct': correct,
            })

        # Field: tgt
        tgt = list(item['src'])
        for mis in mistakes:
            assert tgt[mis['pos']] == mis['wrong']
            tgt[mis['pos']] = mis['correct']
        item['tgt'] = ''.join(tgt)

    def is_letter(c):
        return ord('a') <= ord(c.lower()) <= ord('z')

    # Clean
    for item in tqdm(data, desc='Clean'):
        # Letter
        assert not any(is_letter(c) for c in item['src'])
        assert not any(is_letter(c) for c in item['tgt'])        

        # Whitespace
        assert not any(c.isspace() for c in item['src'])
        assert not any(c.isspace() for c in item['tgt'])

        # Make sure no English symbols
        for s in r'�．!@#$%^&*()_+=`~\|<>,/?:;\'"':
            assert item['src'].find(s) == -1
            assert item['tgt'].find(s) == -1

        # Make sure end with Chinese punctuation
        if not item['src'][-1] in r'。？！：”':
            item['src'] += '。'
            item['tgt'] += '。'

        # Field: errors
        errors = []
        for i, (a, b) in enumerate(zip(item['src'], item['tgt']), start=1):
            if a != b:
                errors.append((i, b))
        item['errors'] = str(errors)

    return data


def write_data(data, output_path):
    rows = ['\t'.join([item['id'], item['src'], item['tgt'],
                       item['errors']]) for item in data]
    with open(output_path, 'w') as f:
        f.write('\n'.join(rows))


if __name__ == '__main__':
    data = load_train_13('SIGHAN2013/SampleSet/Bakeoff2013_SampleSet_WithError_00001-00350.txt')

    print('#Sents in train:', len(data))
    print('n_sent_err:', sum(len(eval(s['errors'])) > 0 for s in data))
    print('n_err:', sum(len(eval(s['errors'])) for s in data))
    print('Ave len:', sum(len(s['src']) for s in data) / len(data))
    print('max len:', max(len(s['src']) for s in data))
    print('min len:', min(len(s['src']) for s in data))

    write_data(data, 'data/train.sighan13-1.tsv')

    # SIGHAN2013/SampleSet/Bakeoff2013_SampleSet_WithError_00001-00350.txt
    # data/train.sighan13-1.tsv
    # SIGHAN2013/SampleSet/Bakeoff2013_SampleSet_WithoutError_10001-10350.txt
    # data/train.sighan13-2.tsv
    # SIGHAN2014/Training/B1_training.sgml 14
    # data/train.sighan14-1.tsv
    # SIGHAN2014/Training/C1_training.sgml 14
    # data/train.sighan14-2.tsv
    # SIGHAN2015/Training/SIGHAN15_CSC_A2_Training.sgml 15
    # data/train.sighan15-1.tsv
    # SIGHAN2015/Training/SIGHAN15_CSC_B2_Training.sgml 15
    # data/train.sighan15-2.tsv
    # Wang27k/train.sgml
    # data/train.wang27k.tsv
