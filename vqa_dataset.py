# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from torch.utils.data import Dataset
import numpy as np
import lmdb
import pickle
import torch
from transformers import BertTokenizer


class VQADataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = args.max_seq_len

        # annotations
        if args.annotation_path.endswith('.npy'):
            self.annotation_db = np.load(
                args.annotation_path,
                allow_pickle=True
            )[1:]
        else:
            raise TypeError('unknown annotation format.')

        # features
        if args.feature_path.endswith('.lmdb'):
            self._init_feature_db(args.feature_path)
            self.base_dir = args.annotation_path
        else:
            raise TypeError('unknown feature format')

    def _init_feature_db(self, feature_path):
        self.env = lmdb.open(
            feature_path,
            subdir=os.path.isdir(feature_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False, buffers=True) as txn:
            self.image_ids = pickle.loads(txn.get(b'keys'))
            self.image_id_indices = {
                self.image_ids[i]: i for i in range(0, len(self.image_ids))
            }

    def _load_feat(self, feat_path: str, convert_to_tensor: bool = False):
        with self.env.begin(write=False, buffers=True) as txn:
            feat = pickle.loads(txn.get(feat_path.encode()))
            if convert_to_tensor:
                feat = torch.from_numpy(feat)
        return feat

    def _load_text_input(self, question: str):
        text = "[CLS] " + question + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(indexed_tokens)

        padding = (self.max_seq_len - len(indexed_tokens)) * [0]
        indexed_tokens += padding
        segments_ids += padding

        indexed_tokens = torch.tensor(indexed_tokens[:self.max_seq_len])
        segments_ids = torch.tensor(segments_ids[:self.max_seq_len])
        assert indexed_tokens.shape == segments_ids.shape
        return torch.stack([indexed_tokens, segments_ids])

    def __len__(self):
        return len(self.annotation_db)

    def __getitem__(self, idx):
        annotation = self.annotation_db[idx]

        # image features
        image_name = annotation["image_name"]
        features = self._load_feat(image_name)
        annotation.update(features)

        # text features
        text_feat = self._load_text_input(annotation["question_str"])
        return {
            "img_feat": torch.tensor(annotation["features"]),
            "text_feat": text_feat
        }
