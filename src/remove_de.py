'''Remove "地" & "得" from output lable file.
'''

import argparse
from os import pardir
import re


def remove_de(input_path, output_path):
    with open(input_path) as f:
        data = f.read()

    data = re.sub(r'\d+, 地(, )?', '', data)
    data = re.sub(r'\d+, 得(, )?', '', data)
    data = re.sub(r', \n', '\n', data)
    data = re.sub(r'(\d{5})\n', r'\1, 0\n', data)

    with open(output_path, 'w') as f:
        f.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()

    remove_de(
        input_path=args.input_path,
        output_path=args.output_path,
    )

    # remove_de(
    #     input_path='test13/bert-mlm-wwm_v1/lbl_test_42.txt',
    #     output_path='test13/bert-mlm-wwm_v1/lbl_test_42_remove.txt',
    # )
