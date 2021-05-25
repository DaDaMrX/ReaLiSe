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
from __future__ import absolute_import, division, print_function

import os
import copy
import json
import logging
import importlib
import random
import shutil
import unittest
import uuid
import tempfile

import sys

from transformers import is_tf_available, is_torch_available

from .utils import require_tf, slow

if is_tf_available():
    import tensorflow as tf
    import numpy as np
    from transformers import TFPreTrainedModel
    # from transformers.modeling_bert import BertModel, BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP

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

def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if '_range' in key or '_std' in key:
            setattr(configs_no_init, key, 0.0)
    return configs_no_init

class TFCommonTestCases:

    @require_tf
    class TFCommonModelTester(unittest.TestCase):

        model_tester = None
        all_model_classes = ()
        test_torchscript = True
        test_pruning = True
        test_resize_embeddings = True

        def test_initialization(self):
            pass
            # config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # configs_no_init = _config_zero_init(config)
            # for model_class in self.all_model_classes:
            #     model = model_class(config=configs_no_init)
            #     for name, param in model.named_parameters():
            #         if param.requires_grad:
            #             self.assertIn(param.data.mean().item(), [0.0, 1.0],
            #             msg="Parameter {} of model {} seems not properly initialized".format(name, model_class))

        def test_save_load(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                model = model_class(config)
                outputs = model(inputs_dict)

                with TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    model = model_class.from_pretrained(tmpdirname)
                    after_outputs = model(inputs_dict)

                    # Make sure we don't have nans
                    out_1 = after_outputs[0].numpy()
                    out_2 = outputs[0].numpy()
                    out_1 = out_1[~np.isnan(out_1)]
                    out_2 = out_2[~np.isnan(out_2)]
                    max_diff = np.amax(np.abs(out_1 - out_2))
                    self.assertLessEqual(max_diff, 1e-5)

        def test_pt_tf_model_equivalence(self):
            if not is_torch_available():
                return

            import torch
            import transformers

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beggining
                pt_model_class = getattr(transformers, pt_model_class_name)

                config.output_hidden_states = True
                tf_model = model_class(config)
                pt_model = pt_model_class(config)

                # Check we can load pt model in tf and vice-versa with model => model functions
                tf_model = transformers.load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=inputs_dict)
                pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model)

                # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
                pt_model.eval()
                pt_inputs_dict = dict((name, torch.from_numpy(key.numpy()).to(torch.long))
                                      for name, key in inputs_dict.items())
                with torch.no_grad():
                    pto = pt_model(**pt_inputs_dict)
                tfo = tf_model(inputs_dict)
                max_diff = np.amax(np.abs(tfo[0].numpy() - pto[0].numpy()))
                self.assertLessEqual(max_diff, 2e-2)

                # Check we can load pt model in tf and vice-versa with checkpoint => model functions
                with TemporaryDirectory() as tmpdirname:
                    pt_checkpoint_path = os.path.join(tmpdirname, 'pt_model.bin')
                    torch.save(pt_model.state_dict(), pt_checkpoint_path)
                    tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(tf_model, pt_checkpoint_path)

                    tf_checkpoint_path = os.path.join(tmpdirname, 'tf_model.h5')
                    tf_model.save_weights(tf_checkpoint_path)
                    pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path)

                # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
                pt_model.eval()
                pt_inputs_dict = dict((name, torch.from_numpy(key.numpy()).to(torch.long))
                                      for name, key in inputs_dict.items())
                with torch.no_grad():
                    pto = pt_model(**pt_inputs_dict)
                tfo = tf_model(inputs_dict)
                max_diff = np.amax(np.abs(tfo[0].numpy() - pto[0].numpy()))
                self.assertLessEqual(max_diff, 2e-2)

        def test_compile_tf_model(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            input_ids = tf.keras.Input(batch_shape=(2, 2000), name='input_ids', dtype='int32')
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

            for model_class in self.all_model_classes:
                # Prepare our model
                model = model_class(config)

                # Let's load it from the disk to be sure we can use pretrained weights
                with TemporaryDirectory() as tmpdirname:
                    outputs = model(inputs_dict)  # build the model
                    model.save_pretrained(tmpdirname)
                    model = model_class.from_pretrained(tmpdirname)

                outputs_dict = model(input_ids)
                hidden_states = outputs_dict[0]

                # Add a dense layer on top to test intetgration with other keras modules
                outputs = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(hidden_states)

                # Compile extended model
                extended_model = tf.keras.Model(inputs=[input_ids], outputs=[outputs])
                extended_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        def test_keyword_and_dict_args(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                model = model_class(config)
                outputs_dict = model(inputs_dict)

                inputs_keywords = copy.deepcopy(inputs_dict)
                input_ids = inputs_keywords.pop('input_ids')
                outputs_keywords = model(input_ids, **inputs_keywords)

                output_dict = outputs_dict[0].numpy()
                output_keywords = outputs_keywords[0].numpy()

                self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-6)

        def test_attention_outputs(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                config.output_attentions = True
                config.output_hidden_states = False
                model = model_class(config)
                outputs = model(inputs_dict)
                attentions = [t.numpy() for t in outputs[-1]]
                self.assertEqual(model.config.output_attentions, True)
                self.assertEqual(model.config.output_hidden_states, False)
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads,
                    self.model_tester.seq_length,
                    self.model_tester.key_len if hasattr(self.model_tester, 'key_len') else self.model_tester.seq_length])
                out_len = len(outputs)

                # Check attention is always last and order is fine
                config.output_attentions = True
                config.output_hidden_states = True
                model = model_class(config)
                outputs = model(inputs_dict)
                self.assertEqual(out_len+1, len(outputs))
                self.assertEqual(model.config.output_attentions, True)
                self.assertEqual(model.config.output_hidden_states, True)

                attentions = [t.numpy() for t in outputs[-1]]
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads,
                    self.model_tester.seq_length,
                    self.model_tester.key_len if hasattr(self.model_tester, 'key_len') else self.model_tester.seq_length])

        def test_hidden_states_output(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                config.output_hidden_states = True
                config.output_attentions = False
                model = model_class(config)
                outputs = model(inputs_dict)
                hidden_states = [t.numpy() for t in outputs[-1]]
                self.assertEqual(model.config.output_attentions, False)
                self.assertEqual(model.config.output_hidden_states, True)
                self.assertEqual(len(hidden_states), self.model_tester.num_hidden_layers + 1)
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size])

        def test_model_common_attributes(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                model = model_class(config)
                assert isinstance(model.get_input_embeddings(), tf.keras.layers.Layer)
                x = model.get_output_embeddings()
                assert x is None or isinstance(x, tf.keras.layers.Layer)

        def test_determinism(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                model = model_class(config)
                first, second = model(inputs_dict, training=False)[0], model(inputs_dict, training=False)[0]
                self.assertTrue(tf.math.equal(first, second).numpy().all())

        def test_inputs_embeds(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            input_ids = inputs_dict["input_ids"]
            del inputs_dict["input_ids"]

            for model_class in self.all_model_classes:
                model = model_class(config)

                wte = model.get_input_embeddings()
                try:
                    x = wte(input_ids, mode="embedding")
                except:
                    try:
                        x = wte([input_ids], mode="embedding")
                    except:
                        try:
                            x = wte([input_ids, None, None, None], mode="embedding")
                        except:
                            if hasattr(self.model_tester, "embedding_size"):
                                x = tf.ones(input_ids.shape + [self.model_tester.embedding_size], dtype=tf.dtypes.float32)
                            else:
                                x = tf.ones(input_ids.shape + [self.model_tester.hidden_size], dtype=tf.dtypes.float32)
                # ^^ In our TF models, the input_embeddings can take slightly different forms,
                # so we try a few of them.
                # We used to fall back to just synthetically creating a dummy tensor of ones:
                #
                inputs_dict["inputs_embeds"] = x
                outputs = model(inputs_dict)


def ids_tensor(shape, vocab_size, rng=None, name=None, dtype=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    output = tf.constant(values,
                         shape=shape,
                         dtype=dtype if dtype is not None else tf.int32)

    return output


if __name__ == "__main__":
    unittest.main()
