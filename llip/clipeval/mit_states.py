"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import glob


standard_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
)


class MITStates(Dataset):
    """
    MIT States with images of objects in various states
    such as slice tomato or purred tomato
    """

    def __init__(
        self,
        data_dir: str,
        transform=standard_transform,
    ):
        self.data_dir = data_dir
        self.dsetname = f"mit_states"

        self.transform = transform
        self.data_df = self.collect_instances()
        self.classnames = self.data_df["valid_classnames"].unique().tolist()
        self.classnames.sort()

        self.static_img_path_list = self.data_df.index.tolist()

    def collect_instances(self):
        img_paths, valid_classnames, attrs = [], [], []
        adjs_per_noun = dict()
        for adj_noun_path in glob.glob(f"{self.data_dir}/images/*"):
            curr_img_paths = glob.glob(adj_noun_path + "/*")

            adj, noun = adj_noun_path.split(f"{self.data_dir}/images/")[-1].split(" ")

            if adj == "adj":
                adj = "typical"

            img_paths.extend(curr_img_paths)
            valid_classnames.extend([noun] * len(curr_img_paths))
            attrs.extend([adj] * len(curr_img_paths))

            if noun not in adjs_per_noun:
                adjs_per_noun[noun] = []
            adjs_per_noun[noun].append(adj)

        data_df = pd.DataFrame(
            list(zip(img_paths, valid_classnames, attrs)),
            columns=["img_path", "valid_classnames", "attr"],
        )
        data_df = data_df.set_index("img_path")

        return data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, ind: int):
        if type(ind) is int:
            img_path = self.static_img_path_list[ind]
        else:
            img_path = ind

        img = Image.open(img_path)

        img_shape = np.array(img).shape
        if len(img_shape) != 3 or img_shape[2] != 3:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        classname = self.data_df.loc[img_path].valid_classnames
        class_idx = self.classnames.index(classname)
        return img, class_idx

    def caption_gt_subpop(self, classname: str, attr: str) -> str:
        return f"{attr} {classname}"


if __name__ == "__main__":
    states = MITStates()
    print(len(states))
    x, y = states[3]
    print(x.shape)
    print(y)
