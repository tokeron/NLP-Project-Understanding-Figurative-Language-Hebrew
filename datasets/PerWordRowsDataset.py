# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""This is a dataset of songs in hebrew with labels for metaphors"""


import os
import datasets
import pandas as pd
from box import Box
import yaml

# You can copy an official description
_DESCRIPTION = """\

"""

_HOMEPAGE = ""

_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "metaphor_dataset": "",
}


class MetaphorParaDatasetV2(datasets.GeneratorBasedBuilder):
    """This is a dataset of songs in hebrew with labels for metaphors"""
    VERSION = datasets.Version("1.0.2")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="metaphor_dataset", version=VERSION, description="metaphor dataset"),
    ]

    DEFAULT_CONFIG_NAME = "metaphor_dataset"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features(
            {
                "sentence": datasets.Sequence(datasets.Value("string")),
                "word": datasets.Sequence(datasets.Value("string")),
                "label": datasets.Sequence(datasets.Value("int64")),
                "word_idx": datasets.Sequence(datasets.Value("int64")),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
        )

    def _split_generators(self, dl_manager):

        data_dir = '/home/tok/figurative-language/figurative-language-data/prepared_data/'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'train', "train_Rows_word_label.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'test', "test_Rows_word_label.json"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train", "validation_Rows_word_label.json"),
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        df = pd.read_json(filepath)
        for index, row in df.iterrows():
            yield index, {
                "sentence": row['sentence'],
                "word": [row['word']],
                "label": [row['label']],
                "word_idx": [row['word_idx']],
            }
