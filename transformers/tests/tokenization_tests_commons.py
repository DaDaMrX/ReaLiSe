# coding=utf-8
# Copyright 2019 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
from io import open
import tempfile
import shutil
import unittest

if sys.version_info[0] == 2:
    import cPickle as pickle

    class TemporaryDirectory(object):
        """Context manager for tempfile.mkdtemp() so it's usable with "with" statement."""
        def __enter__(self):
            self.name = tempfile.mkdtemp()
            return self.name
        def __exit__(self, exc_type, exc_value, traceback):
            shutil.rmtree(self.name)
else:
    import pickle
    TemporaryDirectory = tempfile.TemporaryDirectory
    unicode = str


class CommonTestCases:

    class CommonTokenizerTester(unittest.TestCase):

        tokenizer_class = None

        def setUp(self):
            self.tmpdirname = tempfile.mkdtemp()

        def tearDown(self):
            shutil.rmtree(self.tmpdirname)

        def get_tokenizer(self, **kwargs):
            raise NotImplementedError

        def get_input_output_texts(self):
            raise NotImplementedError

        def test_tokenizers_common_properties(self):
            tokenizer = self.get_tokenizer()
            attributes_list = ["bos_token", "eos_token", "unk_token", "sep_token",
                                "pad_token", "cls_token", "mask_token"]
            for attr in attributes_list:
                self.assertTrue(hasattr(tokenizer, attr))
                self.assertTrue(hasattr(tokenizer, attr + "_id"))

            self.assertTrue(hasattr(tokenizer, "additional_special_tokens"))
            self.assertTrue(hasattr(tokenizer, 'additional_special_tokens_ids'))

            attributes_list = ["max_len", "init_inputs", "init_kwargs", "added_tokens_encoder",
                                "added_tokens_decoder"]
            for attr in attributes_list:
                self.assertTrue(hasattr(tokenizer, attr))

        def test_save_and_load_tokenizer(self):
            # safety check on max_len default value so we are sure the test works
            tokenizer = self.get_tokenizer()
            self.assertNotEqual(tokenizer.max_len, 42)

            # Now let's start the test
            tokenizer = self.get_tokenizer(max_len=42)

            before_tokens = tokenizer.encode(u"He is very happy, UNwant\u00E9d,running", add_special_tokens=False)

            with TemporaryDirectory() as tmpdirname:
                tokenizer.save_pretrained(tmpdirname)
                tokenizer = self.tokenizer_class.from_pretrained(tmpdirname)

                after_tokens = tokenizer.encode(u"He is very happy, UNwant\u00E9d,running", add_special_tokens=False)
                self.assertListEqual(before_tokens, after_tokens)

                self.assertEqual(tokenizer.max_len, 42)
                tokenizer = self.tokenizer_class.from_pretrained(tmpdirname, max_len=43)
                self.assertEqual(tokenizer.max_len, 43)

        def test_pickle_tokenizer(self):
            tokenizer = self.get_tokenizer()
            self.assertIsNotNone(tokenizer)

            text = u"Munich and Berlin are nice cities"
            subwords = tokenizer.tokenize(text)

            with TemporaryDirectory() as tmpdirname:

                filename = os.path.join(tmpdirname, u"tokenizer.bin")
                with open(filename, "wb") as handle:
                    pickle.dump(tokenizer, handle)

                with open(filename, "rb") as handle:
                    tokenizer_new = pickle.load(handle)

            subwords_loaded = tokenizer_new.tokenize(text)

            self.assertListEqual(subwords, subwords_loaded)

        def test_added_tokens_do_lower_case(self):
            tokenizer = self.get_tokenizer(do_lower_case=True)

            special_token = tokenizer.all_special_tokens[0]

            text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
            text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

            toks0 = tokenizer.tokenize(text)  # toks before adding new_toks

            new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd", 'AAAAA BBBBBB', 'CCCCCCCCCDDDDDDDD']
            added = tokenizer.add_tokens(new_toks)
            self.assertEqual(added, 2)

            toks = tokenizer.tokenize(text)
            toks2 = tokenizer.tokenize(text2)

            self.assertEqual(len(toks), len(toks2))
            self.assertNotEqual(len(toks), len(toks0))  # toks0 should be longer
            self.assertListEqual(toks, toks2)

            tokenizer = self.get_tokenizer(do_lower_case=False)

            added = tokenizer.add_tokens(new_toks)
            self.assertEqual(added, 4)

            toks = tokenizer.tokenize(text)
            toks2 = tokenizer.tokenize(text2)

            self.assertEqual(len(toks), len(toks2))  # Length should still be the same
            self.assertNotEqual(len(toks), len(toks0))
            self.assertNotEqual(toks[1], toks2[1])  # But at least the first non-special tokens should differ

        def test_add_tokens_tokenizer(self):
            tokenizer = self.get_tokenizer()

            vocab_size = tokenizer.vocab_size
            all_size = len(tokenizer)

            self.assertNotEqual(vocab_size, 0)
            self.assertEqual(vocab_size, all_size)

            new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
            added_toks = tokenizer.add_tokens(new_toks)
            vocab_size_2 = tokenizer.vocab_size
            all_size_2 = len(tokenizer)

            self.assertNotEqual(vocab_size_2, 0)
            self.assertEqual(vocab_size, vocab_size_2)
            self.assertEqual(added_toks, len(new_toks))
            self.assertEqual(all_size_2, all_size + len(new_toks))

            tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)
            out_string = tokenizer.decode(tokens)

            self.assertGreaterEqual(len(tokens), 4)
            self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
            self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

            new_toks_2 = {'eos_token': ">>>>|||<||<<|<<",
                          'pad_token': "<<<<<|||>|>>>>|>"}
            added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
            vocab_size_3 = tokenizer.vocab_size
            all_size_3 = len(tokenizer)

            self.assertNotEqual(vocab_size_3, 0)
            self.assertEqual(vocab_size, vocab_size_3)
            self.assertEqual(added_toks_2, len(new_toks_2))
            self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

            tokens = tokenizer.encode(">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l",
                                      add_special_tokens=False)
            out_string = tokenizer.decode(tokens)

            self.assertGreaterEqual(len(tokens), 6)
            self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
            self.assertGreater(tokens[0], tokens[1])
            self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
            self.assertGreater(tokens[-2], tokens[-3])
            self.assertEqual(tokens[0], tokenizer.eos_token_id)
            self.assertEqual(tokens[-2], tokenizer.pad_token_id)

        def test_add_special_tokens(self):
            tokenizer = self.get_tokenizer()
            input_text, output_text = self.get_input_output_texts()

            special_token = "[SPECIAL TOKEN]"

            tokenizer.add_special_tokens({"cls_token": special_token})
            encoded_special_token = tokenizer.encode(special_token, add_special_tokens=False)
            assert len(encoded_special_token) == 1

            text = " ".join([input_text, special_token, output_text])
            encoded = tokenizer.encode(text, add_special_tokens=False)

            input_encoded = tokenizer.encode(input_text, add_special_tokens=False)
            output_encoded = tokenizer.encode(output_text, add_special_tokens=False)
            special_token_id = tokenizer.encode(special_token, add_special_tokens=False)
            assert encoded == input_encoded + special_token_id + output_encoded

            decoded = tokenizer.decode(encoded, skip_special_tokens=True)
            assert special_token not in decoded

        def test_required_methods_tokenizer(self):
            tokenizer = self.get_tokenizer()
            input_text, output_text = self.get_input_output_texts()

            tokens = tokenizer.tokenize(input_text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            ids_2 = tokenizer.encode(input_text, add_special_tokens=False)
            self.assertListEqual(ids, ids_2)

            tokens_2 = tokenizer.convert_ids_to_tokens(ids)
            text_2 = tokenizer.decode(ids)

            self.assertEqual(text_2, output_text)

            self.assertNotEqual(len(tokens_2), 0)
            self.assertIsInstance(text_2, (str, unicode))

        def test_encode_decode_with_spaces(self):
            tokenizer = self.get_tokenizer()

            new_toks = ['[ABC]', '[DEF]', 'GHI IHG']
            tokenizer.add_tokens(new_toks)
            input = "[ABC] [DEF] [ABC] GHI IHG [DEF]"
            encoded = tokenizer.encode(input, add_special_tokens=False)
            decoded = tokenizer.decode(encoded)
            self.assertEqual(decoded, input)

        def test_pretrained_model_lists(self):
            weights_list = list(self.tokenizer_class.max_model_input_sizes.keys())
            weights_lists_2 = []
            for file_id, map_list in self.tokenizer_class.pretrained_vocab_files_map.items():
                weights_lists_2.append(list(map_list.keys()))

            for weights_list_2 in weights_lists_2:
                self.assertListEqual(weights_list, weights_list_2)

        def test_mask_output(self):
            if sys.version_info <= (3, 0):
                return

            tokenizer = self.get_tokenizer()

            if tokenizer.build_inputs_with_special_tokens.__qualname__.split('.')[0] != "PreTrainedTokenizer":
                seq_0 = "Test this method."
                seq_1 = "With these inputs."
                information = tokenizer.encode_plus(seq_0, seq_1, add_special_tokens=True)
                sequences, mask = information["input_ids"], information["token_type_ids"]
                self.assertEqual(len(sequences), len(mask))

        def test_number_of_added_tokens(self):
            tokenizer = self.get_tokenizer()

            seq_0 = "Test this method."
            seq_1 = "With these inputs."

            sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=False)
            attached_sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=True)

            # Method is implemented (e.g. not GPT-2)
            if len(attached_sequences) != 2:
                self.assertEqual(tokenizer.num_added_tokens(pair=True), len(attached_sequences) - len(sequences))

        def test_maximum_encoding_length_single_input(self):
            tokenizer = self.get_tokenizer()

            seq_0 = "This is a sentence to be encoded."
            stride = 2

            sequence = tokenizer.encode(seq_0, add_special_tokens=False)
            num_added_tokens = tokenizer.num_added_tokens()
            total_length = len(sequence) + num_added_tokens
            information = tokenizer.encode_plus(seq_0,
                                                max_length=total_length - 2,
                                                add_special_tokens=True,
                                                stride=stride,
                                                return_overflowing_tokens=True)

            truncated_sequence = information["input_ids"]
            overflowing_tokens = information["overflowing_tokens"]

            self.assertEqual(len(overflowing_tokens), 2 + stride)
            self.assertEqual(overflowing_tokens, sequence[-(2 + stride):])
            self.assertEqual(len(truncated_sequence), total_length - 2)
            self.assertEqual(truncated_sequence, tokenizer.build_inputs_with_special_tokens(sequence[:-2]))

        def test_maximum_encoding_length_pair_input(self):
            tokenizer = self.get_tokenizer()

            seq_0 = "This is a sentence to be encoded."
            seq_1 = "This is another sentence to be encoded."
            stride = 2

            sequence_0_no_special_tokens = tokenizer.encode(seq_0, add_special_tokens=False)
            sequence_1_no_special_tokens = tokenizer.encode(seq_1, add_special_tokens=False)

            sequence = tokenizer.encode(seq_0, seq_1, add_special_tokens=True)
            truncated_second_sequence = tokenizer.build_inputs_with_special_tokens(
                tokenizer.encode(seq_0, add_special_tokens=False),
                tokenizer.encode(seq_1, add_special_tokens=False)[:-2]
            )

            information = tokenizer.encode_plus(seq_0, seq_1, max_length=len(sequence) - 2, add_special_tokens=True,
                                                stride=stride, truncation_strategy='only_second',
                                                return_overflowing_tokens=True)
            information_first_truncated = tokenizer.encode_plus(seq_0, seq_1, max_length=len(sequence) - 2,
                                                                add_special_tokens=True, stride=stride,
                                                                truncation_strategy='only_first',
                                                                return_overflowing_tokens=True)

            truncated_sequence = information["input_ids"]
            overflowing_tokens = information["overflowing_tokens"]
            overflowing_tokens_first_truncated = information_first_truncated["overflowing_tokens"]

            self.assertEqual(len(overflowing_tokens), 2 + stride)
            self.assertEqual(overflowing_tokens, sequence_1_no_special_tokens[-(2 + stride):])
            self.assertEqual(overflowing_tokens_first_truncated, sequence_0_no_special_tokens[-(2 + stride):])
            self.assertEqual(len(truncated_sequence), len(sequence) - 2)
            self.assertEqual(truncated_sequence, truncated_second_sequence)

        def test_encode_input_type(self):
            tokenizer = self.get_tokenizer()

            sequence = "Let's encode this sequence"

            tokens = tokenizer.tokenize(sequence)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            formatted_input = tokenizer.encode(sequence, add_special_tokens=True)

            self.assertEqual(tokenizer.encode(tokens, add_special_tokens=True), formatted_input)
            self.assertEqual(tokenizer.encode(input_ids, add_special_tokens=True), formatted_input)

        def test_special_tokens_mask(self):
            tokenizer = self.get_tokenizer()

            sequence_0 = "Encode this."
            sequence_1 = "This one too please."

            # Testing single inputs
            encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
            encoded_sequence_dict = tokenizer.encode_plus(sequence_0, add_special_tokens=True, return_special_tokens_mask=True)
            encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
            special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
            self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

            filtered_sequence = [(x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)]
            filtered_sequence = [x for x in filtered_sequence if x is not None]
            self.assertEqual(encoded_sequence, filtered_sequence)

            # Testing inputs pairs
            encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False) + tokenizer.encode(sequence_1,
                                                                                                         add_special_tokens=False)
            encoded_sequence_dict = tokenizer.encode_plus(sequence_0, sequence_1, add_special_tokens=True,
                                                          return_special_tokens_mask=True)
            encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
            special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
            self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

            filtered_sequence = [(x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)]
            filtered_sequence = [x for x in filtered_sequence if x is not None]
            self.assertEqual(encoded_sequence, filtered_sequence)

            # Testing with already existing special tokens
            if tokenizer.cls_token_id == tokenizer.unk_token_id and tokenizer.cls_token_id == tokenizer.unk_token_id:
                tokenizer.add_special_tokens({'cls_token': '</s>', 'sep_token': '<s>'})
            encoded_sequence_dict = tokenizer.encode_plus(sequence_0,
                                                          add_special_tokens=True,
                                                          return_special_tokens_mask=True)
            encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
            special_tokens_mask_orig = encoded_sequence_dict["special_tokens_mask"]
            special_tokens_mask = tokenizer.get_special_tokens_mask(encoded_sequence_w_special, already_has_special_tokens=True)
            self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))
            self.assertEqual(special_tokens_mask_orig, special_tokens_mask)

        def test_padding_to_max_length(self):
            tokenizer = self.get_tokenizer()

            sequence = "Sequence"
            padding_size = 10
            padding_idx = tokenizer.pad_token_id

            # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
            tokenizer.padding_side = "right"
            encoded_sequence = tokenizer.encode(sequence)
            sequence_length = len(encoded_sequence)
            padded_sequence = tokenizer.encode(sequence, max_length=sequence_length + padding_size, pad_to_max_length=True)
            padded_sequence_length = len(padded_sequence)
            assert sequence_length + padding_size == padded_sequence_length
            assert encoded_sequence + [padding_idx] * padding_size == padded_sequence

            # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
            tokenizer.padding_side = "left"
            encoded_sequence = tokenizer.encode(sequence)
            sequence_length = len(encoded_sequence)
            padded_sequence = tokenizer.encode(sequence, max_length=sequence_length + padding_size, pad_to_max_length=True)
            padded_sequence_length = len(padded_sequence)
            assert sequence_length + padding_size == padded_sequence_length
            assert [padding_idx] * padding_size + encoded_sequence == padded_sequence

            # RIGHT & LEFT PADDING - Check that nothing is done when a maximum length is not specified
            encoded_sequence = tokenizer.encode(sequence)
            sequence_length = len(encoded_sequence)

            tokenizer.padding_side = "right"
            padded_sequence_right = tokenizer.encode(sequence, pad_to_max_length=True)
            padded_sequence_right_length = len(padded_sequence_right)

            tokenizer.padding_side = "left"
            padded_sequence_left = tokenizer.encode(sequence, pad_to_max_length=True)
            padded_sequence_left_length = len(padded_sequence_left)

            assert sequence_length == padded_sequence_right_length
            assert encoded_sequence == padded_sequence_right
            assert sequence_length == padded_sequence_left_length
            assert encoded_sequence == padded_sequence_left

        def test_encode_plus_with_padding(self):
            tokenizer = self.get_tokenizer()

            sequence = "Sequence"
            padding_size = 10
            padding_idx = tokenizer.pad_token_id
            token_type_padding_idx = tokenizer.pad_token_type_id

            encoded_sequence = tokenizer.encode_plus(sequence, return_special_tokens_mask=True)
            input_ids = encoded_sequence['input_ids']
            token_type_ids = encoded_sequence['token_type_ids']
            attention_mask = encoded_sequence['attention_mask']
            special_tokens_mask = encoded_sequence['special_tokens_mask']
            sequence_length = len(input_ids)

            # Test right padding
            tokenizer.padding_side = "right"
            padded_sequence = tokenizer.encode_plus(sequence, max_length=sequence_length + padding_size, pad_to_max_length=True, return_special_tokens_mask=True)
            padded_input_ids = padded_sequence['input_ids']
            padded_token_type_ids = padded_sequence['token_type_ids']
            padded_attention_mask = padded_sequence['attention_mask']
            padded_special_tokens_mask = padded_sequence['special_tokens_mask']
            padded_sequence_length = len(padded_input_ids)

            assert sequence_length + padding_size == padded_sequence_length
            assert input_ids + [padding_idx] * padding_size == padded_input_ids
            assert token_type_ids + [token_type_padding_idx] * padding_size == padded_token_type_ids
            assert attention_mask + [0] * padding_size == padded_attention_mask 
            assert special_tokens_mask + [1] * padding_size == padded_special_tokens_mask 

            # Test left padding
            tokenizer.padding_side = "left"
            padded_sequence = tokenizer.encode_plus(sequence, max_length=sequence_length + padding_size, pad_to_max_length=True, return_special_tokens_mask=True)
            padded_input_ids = padded_sequence['input_ids']
            padded_token_type_ids = padded_sequence['token_type_ids']
            padded_attention_mask = padded_sequence['attention_mask']
            padded_special_tokens_mask = padded_sequence['special_tokens_mask']
            padded_sequence_length = len(padded_input_ids)

            assert sequence_length + padding_size == padded_sequence_length
            assert [padding_idx] * padding_size + input_ids == padded_input_ids
            assert [token_type_padding_idx] * padding_size + token_type_ids == padded_token_type_ids
            assert [0] * padding_size + attention_mask == padded_attention_mask 
            assert [1] * padding_size + special_tokens_mask == padded_special_tokens_mask 