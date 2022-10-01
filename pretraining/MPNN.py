import torch
import torch.nn as nn
from .featurization import get_atom_fdim, mol2graph, index_select_ND

class MPNN(nn.Module):

    def __init__(self, args):

        super(MPNN, self).__init__()

        self.args = args
        self.atom_fdim = get_atom_fdim()
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.num_layer = args.num_layer
        self.dropout = args.dropout  # 0.3
        self.layers_per_message = 1
        self.n_message_layer = 1

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        input_dim = self.atom_fdim  # 133
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        self.W_h = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)  # 300->300
                                  for _ in range(self.n_message_layer)])
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        self.ffn_i = nn.Linear(args.ffn_input, args.ffn_hidden1)
        ffn = []
        ffn.extend([self.act_func,
                    self.dropout_layer,
                    nn.Linear(args.ffn_hidden1, args.ffn_hidden2)]
        )
        ffn.extend([self.act_func,
                    self.dropout_layer,
                    nn.Linear(args.ffn_hidden2, args.ffn_output)]
                   )
        self.ffn = nn.Sequential(*ffn)

    def forward(self, drug_smiles, batch_edges = None):

        mol_graph = mol2graph(self.args, drug_smiles)
        f_atoms, a_neighbors, a_scope = mol_graph.get_components()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, a_neighbors = f_atoms.cuda(), a_neighbors.cuda()

        input = self.W_i(f_atoms)
        message = self.act_func(input)

        nei_a_message = index_select_ND(message, a_neighbors)  # num_atoms x max_num_bonds x hidden
        # 相当于将a_neighbor([[a1,a6,a3],[a2,a5,a4]])中的边index替换为对应的f_atom
        message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        # 例如：将a2对应的边[a1,a6,a3]的特征加起来，达到收集a2邻域信息的效果
        for step in range(self.n_message_layer):
            message = self.W_h[step](message)
            message = self.act_func(input + message)
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a_neighbors
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # mol_vecs得到了一个batch中的所有分子表示
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                # 不计算权重，直接一个分子中的各个原子的特征相加后取平均
                mol_vecs.append(mol_vec)

        output = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        if batch_edges == None:
            return output

        # 根据邻接边，contenate drugs,进行预测
        entity1 = output[batch_edges[0]]
        entity2 = output[batch_edges[1]]

        add = entity1 + entity2
        product = entity1 * entity2
        concatenate = torch.cat((entity1, entity2), dim=1)
        feature = torch.cat((add, product, concatenate), dim=1)

        feat = self.dropout_layer(feature)
        fused_feat = self.ffn_i(feat)
        preds = self.ffn(fused_feat)
        preds = self.sigmoid(preds).view(-1)

        return preds
