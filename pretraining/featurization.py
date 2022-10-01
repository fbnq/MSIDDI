from argparse import Namespace
from rdkit import Chem
import torch

MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2],
}
def get_atom_fdim():
    # Atom feature sizes
    ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
    return ATOM_FDIM

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def atom_features(atom, functional_groups = None):
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features

class MolGraph:

    def __init__(self, smiles):
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.f_atoms = []  # mapping from atom index to atom features
        self.a_neighbor = []

        mol = Chem.MolFromSmiles(smiles)
        self.mol = mol
        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a_neighbor.append([])

        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue

                self.a_neighbor[a1].append(a2)
                self.a_neighbor[a2].append(a1)

class BatchMolGraph:

    def __init__(self, mol_graphs):

        self.mol_graphs = mol_graphs
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim()

        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        a_neighbors = [[]]  # mapping from atom to their neighbors
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)

            for a in range(mol_graph.n_atoms):
                a_neighbors.append([atom + self.n_atoms for atom in mol_graph.a_neighbor[a]])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            # 索引范围从1->n_atoms
            self.n_atoms += mol_graph.n_atoms

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a_neighbors)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.f_atoms = torch.FloatTensor(f_atoms)
        # 包含了一个batch中所有分子的所有原子特征
        self.a_neighbors = torch.LongTensor([a_neighbors[a] + [0] * (self.max_num_bonds - len(a_neighbors[a])) for a in range(self.n_atoms)])
        # 每个原子连接的边数可能不同，为了统一维度，按最大边数补充0值

    def get_components(self):

        return self.f_atoms, self.a_neighbors, self.a_scope

# Memoization
SMILES_TO_GRAPH = {}
def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}
def mol2graph(args, smiles_batch):

    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles)
            SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)
    clear_cache()
    return BatchMolGraph(mol_graphs)

def index_select_ND(source, index):

    index_size = index.size()  # (num_atoms, max_num_bonds)
    # index是a_neighbors
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms * max_num_bonds, hidden_size)
    # index.view(-1)：num_atoms * max_num_bonds
    target = target.view(final_size)  # (num_atoms, max_num_bonds, hidden_size)
    # 相当于将a_neighbor([[a1,a6,a3],[a2,a5,a4]])中的边index替换为对应的f_atom

    return target