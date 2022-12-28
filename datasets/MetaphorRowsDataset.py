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

dataset_type = training_args.data_args.dataset_type
if dataset_type == "pinchasov":
    ending = dataset_type
    sub_version = 41
elif dataset_type == "all":
    ending = dataset_type
    sub_version = 42
elif dataset_type == "pre_piyyut":
    ending = ""
    sub_version = 43
else:
    raise ValueError("Unknown dataset type")

class MetaphorRowsDataset(datasets.GeneratorBasedBuilder):
    """This is a dataset of songs in hebrew with labels for metaphors"""
    VERSION = datasets.Version("2.10.{}".format(sub_version))

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="metaphor_dataset", version=VERSION, description="metaphor dataset"),
    ]

    DEFAULT_CONFIG_NAME = "metaphor_dataset"

    def _info(self):
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
                    "filepath": os.path.join(data_dir, 'train', "train_3_labels{}.json".format(ending)),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, 'test', "test_3_labels{}.json".format(ending)),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train", "validation_3_labels{}.json".format(ending)),
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        df = pd.read_json(filepath)
        for index, row in df.iterrows():
            yield index, {
                "data": row.sentence,
                "labels": row.label,
            }

