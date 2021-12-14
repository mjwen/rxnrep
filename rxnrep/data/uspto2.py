"""
Dataset and datamodule to use other reaction encoder.

This can be used for uspto (tpl, schneider) and grambow dataset, as long as the data
is in tsv format and can be read using `read_uspto_file(filename)`.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from drfp import DrfpEncoder
from pytorch_lightning import LightningDataModule
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer,
)
from sklearn.utils import class_weight
from torch.utils.data import DataLoader

from rxnrep.core.reaction import Reaction
from rxnrep.data.uspto import read_uspto_file

logger = logging.getLogger(__name__)


class AtomPairsFeaturizer:
    def __call__(self, reactions: List[Reaction]) -> List[torch.Tensor]:

        all_feats = []
        for rxn in reactions:
            rct_feats = torch.sum(
                torch.stack([self.featurize_a_mol(m.rdkit_mol) for m in rxn.reactants]),
                dim=0,
            )
            prdt_feats = torch.sum(
                torch.stack([self.featurize_a_mol(m.rdkit_mol) for m in rxn.products]),
                dim=0,
            )

            feats = prdt_feats - rct_feats

            all_feats.append(feats)

        return all_feats

    def featurize_a_mol(self, mol: Chem.Mol) -> torch.Tensor:
        feats = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol,
            minLength=1,
            maxLength=3,
            nBits=256,
        )
        feats = torch.from_numpy(np.asarray(feats, dtype=np.float32))

        return feats


class SmilesDataset:
    """
    Reaction dataset with fingerprints from SMILES.

    Args:
        featurizer: name of the underlying featurizer. Currently supported are:
            `drfp`: https://github.com/reymond-group/drfp
            `rxnfp`:
            `ap3`

    """

    def __init__(
        self,
        filename: Union[str, Path],
        featurizer: str = "drfp",
        num_processes: int = 1,
        has_class_label: bool = True,
    ):
        self.has_class_label = has_class_label
        self.nprocs = num_processes

        self.featurizer = featurizer

        if featurizer == "drfp":
            self.featurizer_fn = DrfpEncoder.encode
        elif featurizer == "rxnfp":
            model, tokenizer = get_default_model_and_tokenizer()
            rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
            self.featurizer_fn = rxnfp_generator.convert
        elif featurizer == "ap3":
            self.featurizer_fn = AtomPairsFeaturizer()

        # read input files
        self.reactions, self._failed = self.read_file(filename)

        # get features
        self.features = self.featurize_reaction()

        # generate labels
        self.labels = [{} for _ in range(len(self.reactions))]
        self.generate_labels()

    @property
    def feature_size(self):
        return {"reaction": len(self.features[0])}

    @property
    def feature_name(self):
        return {"reaction": "reaction features"}

    def featurize_reaction(self):
        if self.featurizer == "drfp":
            smiles = [str(rxn) for rxn in self.reactions]
            features = self.featurizer_fn(smiles)
            features = np.asarray(features, dtype=np.float32)
        elif self.featurizer == "rxnfp":
            smiles = [str(rxn) for rxn in self.reactions]
            features = [self.featurizer_fn(s) for s in smiles]
            features = np.asarray(features, dtype=np.float32)
        elif self.featurizer == "ap3":
            features = self.featurizer_fn(self.reactions)
        else:
            raise ValueError

        return features

    def __len__(self) -> int:
        return len(self.reactions)

    def __getitem__(self, item):
        feats = self.features[item]
        label = self.labels[item]

        return feats, label

    def read_file(self, filename: Path):
        return read_uspto_file(filename, self.nprocs)

    def generate_labels(self):
        """
        Create `reaction_type` label.
        """
        if self.has_class_label:
            for i, rxn in enumerate(self.reactions):
                rxn_class = rxn.get_property("reaction_type")
                self.labels[i]["reaction_type"] = torch.as_tensor(
                    int(rxn_class), dtype=torch.int64
                )

    def get_class_weight(
        self, num_reaction_classes: int = None, class_weight_as_1: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        Args:
            num_reaction_classes: number of reaction classes in the dataset. The class
            labels should be 0, 1, 2, ... num_reaction_classes-1.
            class_weight_as_1: If `True`, the weight for all classes is set to 1.0;
                otherwise, it is inversely proportional to the number of data points in
                the dataset
        """

        if self.has_class_label:
            if class_weight_as_1:
                w = torch.ones(num_reaction_classes)
            else:
                rxn_classes = [rxn.get_property("label") for rxn in self.reactions]

                # class weight for reaction classes
                w = class_weight.compute_class_weight(
                    "balanced",
                    classes=list(range(num_reaction_classes)),
                    y=rxn_classes,
                )
                w = torch.as_tensor(w, dtype=torch.float32)

            weight = {"reaction_type": w}
        else:
            weight = {}

        return weight


class SmilesDataModule(LightningDataModule):
    """
    Uspto datamodule using Morgan feats for classification.
    """

    def __init__(
        self,
        trainset_filename: Union[str, Path],
        valset_filename: Union[str, Path],
        testset_filename: Union[str, Path],
        num_reaction_classes: int,
        featurizer: str = "drfp",
        batch_size: int = 100,
        num_workers: int = 0,
        pin_memory: bool = True,
        num_processes: int = 1,
    ):
        super().__init__()

        self.trainset_filename = trainset_filename
        self.valset_filename = valset_filename
        self.testset_filename = testset_filename

        self.num_reaction_classes = num_reaction_classes

        self.featurizer = featurizer

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_processes = num_processes

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: Optional[str] = None):

        self.data_train = SmilesDataset(
            filename=self.trainset_filename,
            featurizer=self.featurizer,
            num_processes=self.num_processes,
        )

        self.data_val = SmilesDataset(
            filename=self.valset_filename,
            featurizer=self.featurizer,
            num_processes=self.num_processes,
        )

        self.data_test = SmilesDataset(
            filename=self.testset_filename,
            featurizer=self.featurizer,
            num_processes=self.num_processes,
        )

        logger.info(
            f"Trainset size: {len(self.data_train)}, valset size: {len(self.data_val)}: "
            f"testset size: {len(self.data_test)}."
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def get_to_model_info(self) -> Dict[str, Any]:
        class_weight = self.data_train.get_class_weight(
            num_reaction_classes=self.num_reaction_classes, class_weight_as_1=True
        )

        d = {
            "feature_size": self.data_train.feature_size,
            "num_reaction_classes": self.num_reaction_classes,
            "reaction_class_weight": class_weight["reaction_type"],
        }

        return d
