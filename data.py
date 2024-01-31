"""
Vocabluary, dataset, and dataloader objects to support train.py.
"""

import os
import yaml
import torch
from tqdm import tqdm # progress bar for loading data
from torch.utils.data import Dataset, DataLoader

def make(cfg):
    """
    Construct a vocabulary using a presaved ix_to_string file.
    Returns a torch dataloader object.
    """
    vocab = Vocabulary(cfg["data_f"], cfg["smiles_vocab_f"])
    dataset = SMILESDataset(vocab, cfg["data_f"], cfg["block_size"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    """
    Collate function for the SMILESDataset.
    batch: comes in from the DataSet as (batch_size, 2, block_size);
    reshape to give us (2, batch_size, block_size) -- one x, one y.
    """
    x, y = zip(*batch)
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y

class Vocabulary(object):
    def __init__(self, data_f="guacamol_v1_all.smiles", smiles_vocab_f="smiles_vocab.yaml"):
        """
        Vocabulary() uses a pregenerated mapping from unique
        characters in the dataset to an index.
        If the vocabulary file is not found, Vocabulary() will
        try to generate a new one from the dataset.

        data_f: The file containing the SMILES strings;
            default = "guacamol_v1_all.smiles"
        smiles_vocab_f: The file containing the vocabulary;
            default = "smiles_vocab.yaml"
        """
        if not os.path.exists(smiles_vocab_f):
            print("Vocabulary not found. Generating...")
            try:
                make_smiles_vocab(data_f)
            except:
                raise Exception("Failed to generate vocabulary.")
        else:
            print("Loading vocabulary from {}...".format(smiles_vocab_f))
            try:
                with open(smiles_vocab_f, "r") as f:
                    self.ctoi = yaml.load(f, Loader=yaml.FullLoader)
                self.itoc = { ix : char for char, ix in self.ctoi.items() } # ix to char
                print("Loaded vocabulary with {} characters.".format(len(self.ctoi)))
            except:
                raise Exception("Failed to load vocabulary.")
        
    def encode(self, smiles):
        """
        Encode a SMILES string as a list of indices,
        with start and end tokens added.

        smiles: 'CCO\n' --> [0, x, x, 1]
        """
        smiles = smiles.strip()
        encoded = [self.ctoi[char] for char in smiles]
        encoded.insert(0, self.ctoi["<START>"])
        encoded.append(self.ctoi["<END>"])
        return encoded
        
    def decode(self, indices):
        """
        Change a list of indices into a string.
        Indices is probably a tensor; handle it.
        """
        if type(indices) == torch.Tensor:
            indices = indices.squeeze()
            # one at a time, please
            assert len(indices.shape) == 1, \
                "Can only decode one sequence at a time."
            indices = indices.tolist()
        decoded = [self.itoc[ix] for ix in indices]
        decoded = "".join(decoded)
        return decoded

class SMILESDataset(Dataset):
    def __init__(self, vocab, data_f="guacamol_v1_all.smiles", block_size=128):
        """
        This Torch Dataset loads raw SMILES strings from a file,
        then encodes them using a vocabulary of characters to indices.
        The encoded text is stored as a single string.
        When queried by the DataLoader, the Dataset returns a random
        block of text of length "block_size" from the encoded string.
        """
        self.vocab = vocab
        self.block_size = block_size # number of characters to return
        if os.path.exists(data_f):
            try:
                with open(data_f, "r") as f:
                    self.data = f.readlines()
            except:
                raise Exception(f"Failed to load data from {data_f}.")
        else:
            raise Exception(f"Data file {data_f} not found.")
        # print number of loaded strings, with comma separators
        print(f"Loaded {len(self.data):_} SMILES strings.")
        print("Encoding SMILES strings...")
        # encode the data line by line
        self.data = tqdm([self.vocab.encode(smile) for smile in self.data])
        # flatten the list of lists
        self.data = [ix for smile in self.data for ix in smile]
        print(f"Encoded {len(self.data):_} characters.")
        
    def __len__(self):
        """
        Return the number of blocks in the dataset.
        """
        return len(self.data) - self.block_size - 1 # max index, inc. offset

    def __getitem__(self, ix):
        """
        Return the input (x), a text block of length "block_size",
        and the target (y), a block that is offset by one character.
        """
        x = self.data[ix:ix+self.block_size]
        y = self.data[ix+1:ix+self.block_size+1]
        return x, y

def mapping_from_stringlist(smiles):
    """
    Generate a vocabulary from a list of SMILES strings.
    """
    print("Generating vocabulary...")
    chars = set()
    for smile in smiles:
        smile = smile.strip() # remove cruft
        for char in smile:
            chars.add(char)
    chars = list(chars)
    chars.sort()
    """
    Insert special tokens at the beginning of the mapping.
    """
    chars.insert(0, "<START>") # start of molecule
    chars.insert(1, "<END>") # end of molecule
    mapping = { char : ix for ix, char in enumerate(chars) }
    print("Made vocabulary with {} characters.".format(len(mapping)))
    return mapping

def make_smiles_vocab(smiles_data_f):
    with open(smiles_data_f, "r") as f:
        smiles = f.readlines()
    vocab = mapping_from_stringlist(smiles)
    yaml.dump(vocab, open("smiles_vocab.yaml", "w"))

if __name__ == "__main__":
    """
    Run "python data.py <smiles_data>" to generate a vocabulary:
    { 'char' : ix } for each unique character in the dataset.
    By default, the vocabulary is saved to "smiles_vocab.yaml";
    other functions in this repository look for the vocabulary
    by that name.
    """
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_data_f", type=str)
    args = parser.parse_args()
    smiles_data_f = args.smiles_data_f
    make_smiles_vocab(smiles_data_f)
