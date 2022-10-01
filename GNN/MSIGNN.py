import torch
import torch.nn as nn
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import k_hop_subgraph
from utils.metrics import get_scores_main

class MSIGNN(nn.Module):

    def __init__(self, args, config):
        super(MSIGNN, self).__init__()

        self.args = args
        num_of_head = config["num_heads_per_layer"]
        num_of_features = config["num_features_per_layer"]
        self.bias = args.bias
        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.act_func = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        gat_layers = []
        layer1 = GALayer(args, num_of_head[1], num_of_features[0], num_of_features[1], True, True, True, self.bias)
        layer2 = GALayer(args, num_of_head[2], num_of_head[1] * (num_of_features[0] + num_of_features[1]), num_of_features[2], False, True, False, self.bias)
        gat_layers.append(layer1)
        gat_layers.append(layer2)

        self.gat_net = nn.Sequential(
            *gat_layers
        )

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

    def forward(self, data):
        batch_edges = data[1]
        final_data = self.gat_net(data)
        output = final_data[0]

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

class GALayer(nn.Module):

    def __init__(self, args, num_of_head, num_in_feat, num_out_feat, concat, mask, add_res_connection, bias):
        super(GALayer, self).__init__()
        self.src_nodes_dim = 0
        self.trg_nodes_dim = 1
        self.nodes_dim = 0
        self.head_dim = 1

        self.args = args
        self.num_of_heads = num_of_head
        self.concat = concat
        self.mask = mask
        self.add_res_connection = add_res_connection
        in_nodes_features = num_in_feat
        self.num_out_features = num_out_feat

        self.linear_proj = nn.Linear(in_nodes_features, num_of_head * self.num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_head, self.num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_head, self.num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.num_of_heads * (in_nodes_features + self.num_out_features)))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.num_out_features))
        else:
            self.register_parameter('bias', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=args.dropout)

        self.init_params()

    def init_params(self):

        # 按均分布随机初始化
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, out_nodes_features):

        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * out_nodes_features.shape[2])
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):

        scores_per_edge = scores_per_edge - scores_per_edge.max()
        # 每个值都减去最大值
        exp_scores_per_edge = scores_per_edge.exp()  # softmax
        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
        # (E, NH)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        # 完成softmax
        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        # 将trg_index的维度扩展到与exp_scores_per_edge一样
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # 因为要保证index与target_index对应得上
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        # 将某个target对应的所有edge_score加在一起，得到target_score_sum，用于归一化
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        # edge_index有向
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        # 将边对应的头尾节点分别放入头尾节点索引src_nodes_index、trg_nodes_index中
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)

        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted
    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def forward(self, data):
        in_nodes_features, edge_index, local_embedding = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        # assert edge_index.shape[0] == 2, print(f'Expected edge index with shape=(2,E) got {edge_index.shape}')

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        # [n, head, out_dim] [n, 2, 64]
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        source_index_set = []
        target_index_set = []
        batch_belong_set = []
        if self.mask:
            # 获取目标节点的邻域子图（2-hop,根据网络层数来确定）
            # 将邻域节点(包括目标节点本身)作为source_node，目标节点作为target node
            num = 0
            for node in range(num_of_nodes):
                sub_set = k_hop_subgraph(node, self.args.num_of_hop, edge_index, num_nodes=num_of_nodes)
                # 获取子图中的节点
                nodes_sub = sub_set[0]
                source_index = nodes_sub.tolist()
                source_index.remove(node)
                target_index = list([node for j in range(len(source_index))])
                batch_belong = list([num for j in range(len(source_index))])
                source_index_set.extend(source_index)
                target_index_set.extend(target_index)
                batch_belong_set.extend(batch_belong)
                num = num + 1
            source_index_set = torch.Tensor(source_index_set)
            source_index_set = source_index_set.long()
            target_index_set = torch.Tensor(target_index_set)
            target_index_set = target_index_set.long()
            batch_belong_set = torch.Tensor(batch_belong_set)
            batch_belong_set = batch_belong_set.long().cuda()
            new_edge_index = torch.stack((source_index_set, target_index_set), 0).cuda()

            scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source,
                                                                                               scores_target,
                                                                                               nodes_features_proj,
                                                                                               new_edge_index)

            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)  # 得到了a_ij
            attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, new_edge_index[self.trg_nodes_dim],
                                                                  num_of_nodes)  # shape = (E, NH, 1)

            att_per_edge = torch.div(attentions_per_edge.squeeze(dim=-1).sum(dim = -1), 0.5)
            perm = topk(att_per_edge.squeeze(dim=-1), self.args.ratio, batch_belong_set)

            attentions_per_edge = torch.index_select(attentions_per_edge, 0, perm)
            nodes_features_proj_lifted = torch.index_select(nodes_features_proj_lifted, 0, perm)
            new_edge_index = torch.index_select(new_edge_index, 1, perm)
        else:
            new_edge_index = edge_index
            scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source,
                                                                                               scores_target,
                                                                                               nodes_features_proj,
                                                                                               new_edge_index)
            scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)  # 得到了a_ij
            attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, new_edge_index[self.trg_nodes_dim],
                                                                  num_of_nodes)  # shape = (E, NH, 1)
            attentions_per_edge = self.dropout(attentions_per_edge)


        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, new_edge_index,
                                                      in_nodes_features, num_of_nodes)
        res_connect = local_embedding.view(local_embedding.shape[0], 1, local_embedding.shape[1])
        res_connect = torch.cat((res_connect, res_connect), 1)

        if self.add_res_connection:
            # concate original local embedding
            out_nodes_features = torch.cat((out_nodes_features, res_connect), 2)
            # print(f'shape of out_nodes_features:{out_nodes_features.shape}')
        out_nodes_features = self.skip_concat_bias(out_nodes_features)
        # softmax之后的attention；输入的节点特征；aggregate邻域之后的节点特征

        return (out_nodes_features, edge_index, local_embedding)