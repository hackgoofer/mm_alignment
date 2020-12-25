# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class AlignmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
    
    def forward(self, image_repr, text_repr):
        image_latent = self.image_encoder(image_repr)
        text_latent = self.text_encoder(text_repr)
        return (image_latent, text_latent)

class ImageEncoder(nn.Module):
    def __init__(self, indim=2048, outdim=2048, hiddendim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(indim, hiddendim),
            nn.ReLU(),
            nn.Linear(hiddendim, outdim),
        )

    def forward(self, x):
        return self.encoder(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.concat_n_classifier = nn.Linear(self.model.config.hidden_size * 4, 2048)
          
    def _last_cls_representation(self, output):
        pass

    def _concat_last_n_layers(self, n, output):
        hidden_states = output[2]
        pooled_output = torch.cat(tuple([hidden_states[i] for i in range(-n,0,1)]), dim=-1)
        pooled_output = pooled_output[:, 0, :]
        input_dim = pooled_output.shape[-1]
        logits = self.concat_n_classifier(pooled_output)
        return logits

    def forward(self, x):
        tokens_tensor = x[:, 0, :]
        segments_tensor = x[:, 1, :]
        outputs = self.model(tokens_tensor, token_type_ids=segments_tensor)
        
        # TODO: Try different representation
        latent = self._concat_last_n_layers(4, outputs)
        return latent