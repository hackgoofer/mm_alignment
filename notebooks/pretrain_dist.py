import sys
sys.path.append('../')
import torch
import numpy as np
import argparse
from model import AlignmentModel
from torch.utils.data import DataLoader
from cc_dataset import CCDataset

device = "cuda:1"

model_path = "../checkpoints/2020_12_23_1/epch0_bidx6400.pt"
model = AlignmentModel().to(device)
model_data = torch.load(model_path)["model_state_dict"]
model_data.pop("text_encoder.model.embeddings.position_ids")
model.load_state_dict(model_data)
model.eval()


class Namespace:
    def __init__(self, opts):
        self.__dict__.update(opts)


def collate_fn(batch):
    image_feats = []
    text_feats = []
    for sample in batch:
        image_feats.append(sample["img_feat"])
        text_feats.append(sample["text_feat"])
    image_feats = torch.mean(torch.stack(image_feats), dim=1).to(device)
    text_feats = torch.stack(text_feats).to(device)
    return [image_feats, text_feats]

pretrain_args = argparse.Namespace(
    max_seq_len=16,
    annotation_path='/private/home/sash/.cache/torch/mmf/data/datasets/cc/defaults/annotations/train_all.npy',
    batch_size=32,
    feature_path='/private/home/sash/.cache/torch/mmf/data/datasets/cc/defaults/features/cc_train.lmdb',
)

pretrain_dataset = CCDataset(pretrain_args)
pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=pretrain_args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

with torch.no_grad():
    diag_distribution = torch.tensor([])
    print("Eval on pretrain...")
    for batch_idx, batch in enumerate(pretrain_dataloader):
        image_feat_batch = batch[0]  # B x 2048
        text_feat_batch = batch[1]
        image_repr, text_repr = model(image_feat_batch, text_feat_batch)
        dotproduct = image_repr.mm(text_repr.t())
        diag = dotproduct.detach().cpu().diagonal()
        diag_distribution = torch.cat([diag_distribution, diag])
        
        if batch_idx % 1500 == 0:
            print(f"Saving pretrain diag at {batch_idx}")
            torch.save(diag_distribution, f"/private/home/sash/pretrain_diag/{batch_idx}.pt")
            diag_distribution = torch.tensor([])

        if batch_idx % 100 == 0:
            print(f"Progress: {batch_idx}")

    torch.save(diag_distribution, "/private/home/sash/pretrain_diag/final.pt")
    print(f"Saving pretrain diag at final: {batch_idx}")
