import os.path as osp

from zhijian.data.domain import Datum, DatasetBase


class DomainNet(DatasetBase):
    """DomainNet.

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.

    Special note: the t-shirt class (327) is missing in painting_train.txt.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """

    domains = [
        "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
    ]

    def __init__(self, root, source_domain, target_domain):
        root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = root
        self.split_dir = osp.join(self.dataset_dir, "splits")

        self.check_input_domains(
            source_domain, target_domain
        )

        train_x = self._read_data(source_domain, split="train")
        train_u = self._read_data(target_domain, split="train")
        val = self._read_data(source_domain, split="test")
        test = self._read_data(target_domain, split="test")

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname
                    )
                    items.append(item)

        return items
