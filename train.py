# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from torch.utils.data import DataLoader
from vqa_dataset import VQADataset
from model import AlignmentModel
from loss import ContrastiveLoss
import torch

device = "cuda:1"
max_epochs = 3
ckpt_path = "/private/home/sash/alignment/checkpoints"
tag = "2020_12_23_2"


def collate_fn(batch):
    image_feats = []
    text_feats = []
    for sample in batch:
        image_feats.append(sample["img_feat"])
        text_feats.append(sample["text_feat"])
    image_feats = torch.mean(torch.stack(image_feats), dim=1).to(device)
    text_feats = torch.stack(text_feats).to(device)
    return [image_feats, text_feats]


def get_dataset_loader(args):
    dataset = VQADataset(args)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


def get_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument(
        '--annotation_path',
        help='Supported format: npy',
        type=str,
    )
    parser.add_argument(
        '--feature_path',
        help='Supported format: npy',
        type=str,
    )
    parser.add_argument(
        '--no_tqdm',
        dest='no_tqdm',
        action='store_true',
    )
    parser.add_argument(
        '--bs',
        dest='batch_size',
        default=32,
        type=int,
    )
    parser.add_argument(
        '--msl',
        dest='max_seq_len',
        default=16,
        type=int,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train_data_loader = get_dataset_loader(args)
    model = AlignmentModel().to(device)
    criterion = ContrastiveLoss(margin=0.2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    losses = []
    for epoch_idx in range(max_epochs):
        num_batch_in_epoch = len(train_data_loader)
        for batch_idx, batch in enumerate(train_data_loader):
            model.train()
            optimizer.zero_grad()
            image_feat_batch = batch[0]  # B x 2048
            text_feat_batch = batch[1]
            image_repr, text_repr = model(image_feat_batch, text_feat_batch)
            loss = criterion(image_repr, text_repr)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if batch_idx % 100 == 0 and not (batch_idx == 0 and epoch_idx == 0):
                avg_loss = sum(losses)/len(losses)
                losses = []
                print(f"epoch {epoch_idx}, batch {batch_idx}, loss {avg_loss}")
                torch.save({
                    'epoch': epoch_idx,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, f"{ckpt_path}/{tag}/epch{epoch_idx}_bidx{batch_idx}.pt")

            if (batch_idx + epoch_idx * num_batch_in_epoch) % 800 == 0:
                # eval
                model.eval()
                with torch.no_grad:
                    pass
                pass


if __name__ == "__main__":
    main()