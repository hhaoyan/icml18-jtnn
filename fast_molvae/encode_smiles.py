import pickle
import sys

from tqdm import tqdm

sys.path.append('..')

import rdkit
import torch

from fast_jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

vocab = [x.strip("\r\n ") for x in open('../data/moses/vocab.txt')]
vocab = Vocab(vocab)

model = JTNNVAE(vocab, 450, 56, 20, 3)
model.load_state_dict(torch.load('moses-h450z56/model.iter-400000'))
model = model.cuda()

special_smiles = [
    'CC(=O)NC1=CC=C(C=C1)O',  # Acetaminophen
]

with open('../data/moses/train.txt') as f:
    smiles = [x.strip() for x in f.readlines()]

batches = []
problems = 0
with tqdm(total=len(smiles)) as bar:
    for i in range(0, len(smiles), 4):
        end = min(i + 4, len(smiles))
        try:
            vector = model.encode_from_smiles(smiles[i:end])
            embedding = vector.detach().cpu().numpy()

            batches.append({
                'smiles': smiles[i:end],
                'embedding': embedding
            })
            bar.update(n=end - i)
        except KeyError:
            problems += end - i
            bar.update(end - i)
            bar.set_postfix({
                'problems': problems
            })

with open('moses_embedding.pickle', 'wb') as f:
    pickle.dump(batches, f)
