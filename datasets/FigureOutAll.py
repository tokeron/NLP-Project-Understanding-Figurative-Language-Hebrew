"""This is a dataset of songs in hebrew with labels for metaphors"""


import os
import datasets
import pandas as pd
from box import Box
import yaml
from config.config_parser import *

# You can copy an official description
_DESCRIPTION = """

"""

_HOMEPAGE = ""

_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "metaphor_dataset": "",
}




class MetaphorRowsDataset(datasets.GeneratorBasedBuilder):

    """This is a dataset of songs in hebrew with labels for metaphors"""
    # VERSION = datasets.Version("11{}.{}.{}".format(version, seed, rows_per_example))
    VERSION = datasets.Version("1.1.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="metaphor_dataset", version=VERSION, description="metaphor dataset"),
    ]

    DEFAULT_CONFIG_NAME = "metaphor_dataset"

    def _info(self):
        if dataset_version == 'tok':
            features = datasets.Features(
                {
                    "data": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-metaphor",
                                "I-metaphor",
                            ]
                        )
                    )
                }
            )
        else:
            features = datasets.Features(
                {
                    "data": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Value("int32"),
                    "w_index": datasets.Value("int32"),
                    "word": datasets.Value("string"),
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
                    "filepath": os.path.join(data_dir, 'train', "HP{}_train{}_{}_rows_{}.json".format(
                        dataset_version,
                        "_non_bin" if non_bin else "",
                        seed,
                        rows_per_example)),
                    # "filepath": os.path.join(data_dir, 'train', "train_mixed.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'test', "HP{}_test{}_{}_rows_{}.json".format(
                        dataset_version,
                        "_non_bin" if non_bin else "",
                        seed,
                        rows_per_example)),
                    # "filepath": os.path.join(data_dir, 'test', "test_mixed.json"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train", "HP{}_validation{}_{}_rows_{}.json".format(
                        dataset_version,
                        "_non_bin" if non_bin else "",
                        seed,
                        rows_per_example)),
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        df = pd.read_json(filepath)
        for index, row in df.iterrows():
            if dataset_version == 'tok':
                yield index, {
                    "data": row.sentence,
                    "labels": row.label,
                }
            else:
                yield index, {
                    "data": row.sentence,
                    "labels": row.label,
                    "w_index": row.w_index,
                    "word": row.word,
                }
