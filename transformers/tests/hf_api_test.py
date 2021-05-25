# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
import six
import time
import unittest

from transformers.hf_api import HfApi, S3Obj, PresignedUrl, HfFolder, HTTPError

USER = "__DUMMY_TRANSFORMERS_USER__"
PASS = "__DUMMY_TRANSFORMERS_PASS__"
FILE_KEY = "Test-{}.txt".format(int(time.time()))
FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/input.txt"
)



class HfApiCommonTest(unittest.TestCase):
    _api = HfApi(endpoint="https://moon-staging.huggingface.co")


class HfApiLoginTest(HfApiCommonTest):
    def test_login_invalid(self):
        with self.assertRaises(HTTPError):
            self._api.login(username=USER, password="fake")

    def test_login_valid(self):
        token = self._api.login(username=USER, password=PASS)
        self.assertIsInstance(token, six.string_types)


class HfApiEndpointsTest(HfApiCommonTest):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = cls._api.login(username=USER, password=PASS)

    def test_whoami(self):
        user = self._api.whoami(token=self._token)
        self.assertEqual(user, USER)

    def test_presign(self):
        urls = self._api.presign(token=self._token, filename=FILE_KEY)
        self.assertIsInstance(urls, PresignedUrl)
        self.assertEqual(urls.type, "text/plain")

    def test_presign_and_upload(self):
        access_url = self._api.presign_and_upload(
            token=self._token, filename=FILE_KEY, filepath=FILE_PATH
        )
        self.assertIsInstance(access_url, six.string_types)

    def test_list_objs(self):
        objs = self._api.list_objs(token=self._token)
        self.assertIsInstance(objs, list)
        if len(objs) > 0:
            o = objs[-1]
            self.assertIsInstance(o, S3Obj)



class HfFolderTest(unittest.TestCase):
    def test_token_workflow(self):
        """
        Test the whole token save/get/delete workflow,
        with the desired behavior with respect to non-existent tokens.
        """
        token = "token-{}".format(int(time.time()))
        HfFolder.save_token(token)
        self.assertEqual(
            HfFolder.get_token(),
            token
        )
        HfFolder.delete_token()
        HfFolder.delete_token()
        # ^^ not an error, we test that the
        # second call does not fail.
        self.assertEqual(
            HfFolder.get_token(),
            None
        )


if __name__ == "__main__":
    unittest.main()
