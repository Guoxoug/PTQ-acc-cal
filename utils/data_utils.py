
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import os


TRAIN_DATASETS = ["cifar10", "cifar100", "imagenet"]


# Transforms for when imaged data is loaded -----------------------------------

# taken from https://arxiv.org/pdf/1512.03385.pdf
# also used in https://arxiv.org/abs/1608.06993
# contains transforms for each dataset, norm is factored out
# so that when dataset is OOD, the training values can be used
CIFAR_10_TRANSFORMS = {
    "train": tv.transforms.Compose(
        [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor()
        ]
    ),
    "test": tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
        ]
    ),
    "norm": tv.transforms.Normalize(
        mean=[0.4914, 0.4823, 0.4465],
        std=[0.247, 0.243, 0.261]
    )
}


CIFAR_100_TRANSFORMS = {
    "train": tv.transforms.Compose(
        [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ]
    ),
    "test": tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
        ]
    ),
    "norm": tv.transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    ),
}

# https://github.com/pytorch/vision/issues/39#issuecomment-403701432
# https://paperswithcode.github.io/torchbench/imagenet/
# not necessarily the same as all papers
IMAGENET_TRANSFORMS = {
    "train": tv.transforms.Compose(
        [
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ]
    ),
    "test": tv.transforms.Compose(
        [
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
           
        ]
    ),
    "norm":  tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
}



DATASET_NAME_TRANSFORM_MAPPING = {
    "imagenet": IMAGENET_TRANSFORMS,
    "cifar10": CIFAR_10_TRANSFORMS,
    "cifar100": CIFAR_100_TRANSFORMS
}


# get transforms for pre-processing
def get_preprocessing_transforms(
    dataset_name, id_dataset_name=None
) -> dict:
    """Get preprocessing transforms for a dataset.
    
    If the dataset is OOD then the ID/training set's normalisation values will
    be used (as if preprocessing is part of network input layer).
    """
    dataset_transforms = DATASET_NAME_TRANSFORM_MAPPING[dataset_name]
    if id_dataset_name is None:
        transforms = {
            "train": tv.transforms.Compose(
                [
                    dataset_transforms["train"],
                    dataset_transforms["norm"]
                ]
            ),
            "test": tv.transforms.Compose(
                [
                    dataset_transforms["test"],
                    dataset_transforms["norm"]
                ]
            )
        }
    else:

        # use the in distribution/training set's values for testing ood
        id_dataset_transforms = DATASET_NAME_TRANSFORM_MAPPING[id_dataset_name]
        transforms = {
            "train": tv.transforms.Compose(
                [
                    dataset_transforms["train"],
                    id_dataset_transforms["norm"]
                ]
            ),
            "test": tv.transforms.Compose(
                [
                    dataset_transforms["test"],
                    id_dataset_transforms["norm"]
                ]
            )
        }
    return transforms

# Data object -----------------------------------------------------------------

class Data:
    """Class that contains a datasets + loaders as well as information
    about the dataset, e.g. #samples, transforms for data augmentation.
    Allows the division of the training set into train and validation sets.
    """
    def __init__(
        self,
        name: str, 
        datapath: str,
        download=False,
        batch_size=64,
        test_batch_size=None,
        num_workers=8,
        drop_last=False,
        transforms={"train":None, "test":None},
        target_transforms={"train": None, "test": None},
        val_size=0,
        num_classes=None,
        test_only=False, 
        **data_kwargs
    ) -> None:
        self.name = name
        self.datapath = datapath
        self.download = download
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size if (
            test_batch_size is not None
        ) else batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last 
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.val_size = val_size
        self.num_classes = num_classes # this overwrites defaults
        self.test_only = test_only
        self.data_kwargs = data_kwargs

        # get datasets and dataloaders

        # training/in distribution sets ---------------------------------------
        if "cifar" in self.name and "-c" not in self.name:

            # both datasets are of the same format
            if self.name == "cifar10":
                CIFAR = tv.datasets.CIFAR10
                self.num_classes = 10 if (
                    self.num_classes is None
                ) else self.num_classes
            if self.name == "cifar100":
                CIFAR = tv.datasets.CIFAR100
                self.num_classes = 100 if (
                    self.num_classes is None
                ) else self.num_classes

            # test set/loader
            self.test_set = CIFAR(
                root=self.datapath,
                train=False,
                transform=self.transforms["test"],
                target_transform=self.target_transforms["test"],
                download=self.download
            )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=self.drop_last
            )

            # train
            if not self.test_only:
                self.train_set = CIFAR(
                    root=self.datapath,
                    train=True,
                    transform=self.transforms["train"],
                    target_transform=self.target_transforms["train"],
                    download=self.download
                )

                indices = torch.arange(0, len(self.train_set))

                if val_size > 0:
                    assert (
                        val_size <= len(self.train_set)
                    ), "val size larger than training set"

                    # train set with test transforms
                    self.val_set = CIFAR(
                        root=self.datapath,
                        train=True,
                        transform=self.transforms["test"],
                        target_transform=self.target_transforms["test"],
                        download=self.download
                    )

                    # train/val split
                    self.train_indices = indices[0:-val_size]
                    self.val_indices = indices[val_size:]

                    self.val_loader = DataLoader(
                        self.val_set,
                        batch_size=self.test_batch_size,
                        num_workers=self.num_workers,
                        drop_last=self.drop_last,
                        sampler=SequentialSampler(
                            self.val_indices
                        )
                    )

                else:
                    self.train_indices = indices

                self.train_loader = DataLoader(
                    self.train_set,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=self.drop_last,
                    sampler=SubsetRandomSampler(
                        self.train_indices
                    )
                )
        
        if self.name == "imagenet":

            self.num_classes = 1000 if (
                self.num_classes is None
            ) else self.num_classes

            
            # test set/loader
            # for imagenet use val as test
            # imagenet must be previously downloaded

            # torchvision uses tar.gz files
            try:
                self.test_set = tv.datasets.ImageNet(
                    root=self.datapath,
                    split="val",
                    transform=self.transforms["test"],
                    target_transform=self.target_transforms["test"]
            )

            # unzipped folders
            except:
                self.test_set = tv.datasets.ImageFolder(
                    root=os.path.join(self.datapath, "validation"),
                    transform=self.transforms["test"],
                    target_transform=self.target_transforms["test"]
                )

            self.test_loader = DataLoader(
                self.test_set,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=True
            )

            # train
            if not self.test_only:
                try:
                    self.train_set = tv.datasets.ImageNet(
                        root=self.datapath,
                        split="train",
                        transform=self.transforms["train"],
                        target_transform = self.target_transforms["train"]
                    )
                except:
                    self.train_set = tv.datasets.ImageFolder(
                        root=os.path.join(self.datapath, "train"),
                        transform=self.transforms["train"],
                        target_transform=self.target_transforms["train"]
                    )

                indices = torch.arange(0, len(self.train_set))

                if val_size > 0:
                    assert (
                        val_size <= len(self.train_set)
                    ), "val size larger than training set"

                    # train set with test transforms
                    try:
                        self.val_set = tv.datasets.ImageNet(
                            root=self.datapath,
                            split="train",
                            transform=self.transforms["test"],
                            target_transform=self.target_transforms["test"]
                        )
                    except:
                        self.test_set = tv.datasets.ImageFolder(
                            root=os.path.join(self.datapath, "train"),
                            transform=self.transforms["test"],
                            target_transform=self.target_transforms["test"]
                        )

                    # train/val split
                    self.train_indices = indices[0:-val_size]
                    self.val_indices = indices[val_size:]

                    self.val_loader = DataLoader(
                        self.val_set,
                        batch_size=self.test_batch_size,
                        num_workers=self.num_workers,
                        drop_last=self.drop_last,
                        sampler=SequentialSampler(
                            self.val_indices
                        ),
                        pin_memory=True
                    )

                else:
                    self.train_indices = indices

                self.train_loader = DataLoader(
                    self.train_set,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    drop_last=self.drop_last,
                    sampler=SubsetRandomSampler(
                        self.train_indices
                    ),
                    pin_memory=True
                )




# debugging
if __name__ ==  "__main__":

    imagenet = Data(
        "imagenet",
        "/work/ImageNet/imagenet/manual",
        num_classes=1000,
        transforms=get_preprocessing_transforms(
            "imagenet"),
    )
    print(imagenet.test_set.targets)
    cifar100 = Data(
        "cifar100",
        "/work/cifar",
        num_classes=100,
        transforms=get_preprocessing_transforms(
            "cifar100"
        )
    )

    print(cifar100.transforms)

            

        

        

