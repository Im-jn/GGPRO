import json
import math
import copy

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, GATConv, GATv2Conv
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_len=500, dropout=0.0):
        super(TransDecoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.d_model = d_model
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, tgt, memory=None, tgt_key_padding_mask=None):
        # Embedding and positional encoding
        tgt = tgt * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        # tgt = self.pos_encoder(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        # Decoder forward pass
        if memory is None:  # use self-attention
            output = self.transformer_decoder(tgt, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.fc(output)
        return output


class HGNNs(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, layers_num, gnn_name="gat", dropout=0.0, 
                 node_names="poi", edge_names=("u_trans", "o_trans", "near", "same")):
        '''
        :param node_names: names of node types in the heterogeneous graph
        :param in_feats: size of input feature
        :param hid_feats: size of hidden layer feature
        :param out_feats: size of output feature
        :param layers_num: number of hidden layers
        :param gnn_name: type of GNN to use ("gcn" or "gat")
        :param edge_names: names of edge types in the heterogeneous graph
        '''
        super(HGNNs, self).__init__()

        self.layers_num = layers_num
        self.node_names = node_names
        self.edge_names = edge_names
        self.gnn_name = gnn_name
        self.dropout = dropout
        self.heads_num = 4

        # Define the input, hidden, and output layers
        self.input_layer = self._build_layer(in_feats, hid_feats)
        if layers_num > 2:
            self.hidden_layers = nn.ModuleList([
                self._build_layer(hid_feats, hid_feats) for _ in range(layers_num - 2)
            ])
        else:
            self.hidden_layers = []
        self.output_layer = self._build_layer(hid_feats, out_feats)

    def _build_layer(self, in_feats, out_feats):
        if self.gnn_name == "gat":
            conv_func = lambda in_feats, out_feats: GATConv(
                in_feats, out_feats // self.heads_num, num_heads=self.heads_num,
                feat_drop=self.dropout, attn_drop=self.dropout, residual=True
            )
        else:
            conv_func = GraphConv

        return HeteroGraphConv({
            edge: conv_func(in_feats, out_feats) for edge in self.edge_names
        }, aggregate='sum')

    def _postprocess(self, x):
        if x.dim() == 3:
            x = x.flatten(1)
            return F.elu(x)
        else:
            return F.relu(x)

    def _flatten_heads(self, x):
        if x.dim() == 3:
            x = x.flatten(1)
        return x

    def forward(self, graph, poi_inputs):
        h = self.input_layer(graph, {self.node_names: poi_inputs})
        h = {k: self._postprocess(v) for k, v in h.items()}
        for layer in self.hidden_layers:
            h = layer(graph, h)
            h = {k: self._postprocess(v) for k, v in h.items()}
        outputs = self.output_layer(graph, h)
        outputs = {k: self._flatten_heads(v) for k, v in outputs.items()}
        return outputs[self.node_names]


class EmbedLayer(nn.Module):
    def __init__(self, poi_num, cate_num, embed_dim, boundary_box, fusion_type=["concat", "add"][1]):
        super(EmbedLayer, self).__init__()
        if fusion_type == "concat":
            self.id_embed = nn.Embedding(poi_num, embed_dim//2)
            self.cate_embed = nn.Embedding(cate_num, embed_dim//2)
        else:
            self.id_embed = nn.Embedding(poi_num, embed_dim)
            self.cate_embed = nn.Embedding(cate_num, embed_dim)
        self.fusion_type = fusion_type
        self.boundary_box = boundary_box
        self.loc_enc = LocationEncoder(boundary_box, embed_dim)

    def forward(self, poi_ids, cate_ids, lons, lats):
        id_emb = self.id_embed(poi_ids)
        cate_emb = self.cate_embed(cate_ids)
        if self.fusion_type == "concat":
            loc_emb = self.loc_enc(lons, lats)
            return torch.cat([id_emb, cate_emb], dim=-1) + loc_emb
            # return torch.cat([id_emb, cate_emb], dim=-1)
        else:
            loc_emb = self.loc_enc(lons, lats)
            return (id_emb + cate_emb) + loc_emb
            # return (id_emb + cate_emb)


class LocationEncoder(nn.Module):
    def __init__(self, boundary_box, embed_dim):
        """
        Args:
            boundary_box: (lon_min, lon_max, lat_min, lat_max)
            embed_dim: Output embedding dimension (must be a multiple of 4)
        """
        super(LocationEncoder, self).__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        
        self.boundary_box = boundary_box
        self.embed_dim = embed_dim
        self.freq_dim = embed_dim // 4  # lon-sin, lon-cos, lat-sin, lat-cos
        self.eps = 1e-6

        # Frequency terms (log scale)
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.arange(self.freq_dim, dtype=torch.float32)
                * -(math.log(10000.0) / self.freq_dim)
            )
        )

    def forward(self, lons, lats):
        """
        Args:
            lons, lats: shape (...), any batch shape
        Returns:
            loc_emb: shape (..., embed_dim)
        """
        # Ensure input is float32 to avoid type mismatch errors
        lons = lons.float()
        lats = lats.float()
        
        lon_min, lon_max, lat_min, lat_max = self.boundary_box

        # 1. normalize to [0, 1]
        lon_den = (lon_max - lon_min)
        lat_den = (lat_max - lat_min)
        if lon_den == 0:
            lon_den = self.eps
        if lat_den == 0:
            lat_den = self.eps
        lon = (lons - lon_min) / lon_den
        lat = (lats - lat_min) / lat_den

        # 2. scale to [0, 2π]
        scale = 500.0 * math.pi
        lon = lon.unsqueeze(-1) * scale
        lat = lat.unsqueeze(-1) * scale

        # 3. positional encoding
        lon_enc = lon * self.freqs.to(lon.device)
        lat_enc = lat * self.freqs.to(lat.device)

        lon_sin = torch.sin(lon_enc)
        lon_cos = torch.cos(lon_enc)
        lat_sin = torch.sin(lat_enc)
        lat_cos = torch.cos(lat_enc)

        # 4. concat
        loc_emb = torch.cat(
            [lon_sin, lon_cos, lat_sin, lat_cos],
            dim=-1
        )

        return torch.nan_to_num(loc_emb, nan=0.0, posinf=0.0, neginf=0.0)


class NodesJudger(nn.Module):
    def __init__(self, in_feats, hid_feats, layers_num, poi_num, cate_num, boundary_box, dropout=0.0, 
                 gnn_name="gat", node_name=["poi", "category"][0], edge_names=("u_trans", "o_trans", "near", "same", "recommend")):
        super(NodesJudger, self).__init__()
        self.embed_layer = EmbedLayer(poi_num, cate_num, in_feats, boundary_box)
        self.gnn = HGNNs(in_feats, hid_feats, in_feats, layers_num, gnn_name,
                         dropout=dropout, node_names=node_name, edge_names=edge_names)
        self.traj_model = TransDecoder(in_feats, nhead=8, num_layers=3, dropout=dropout)
        self.out_layer = nn.Linear(in_feats, 1)
        self.dropout = nn.Dropout(dropout)

    def unbatch_g_embed(self, b_graph, g_embed):
        b_graph.ndata["n_value"] = g_embed
        un_b_graphs = dgl.unbatch(b_graph)
        node_embed = []
        for graph in un_b_graphs:
            node_embed.append(graph.ndata["n_value"])
        if len(node_embed) > 1:
            result = pad_sequence(node_embed, batch_first=True)
        else:
            result = node_embed[0]
        del b_graph.ndata['n_value']
        return result

    def forward(self, graph, poi_inputs, traj_inputs, recover=None):
        g_embed = self.embed_layer(poi_inputs[0], poi_inputs[1], poi_inputs[2], poi_inputs[3])
        # Get the node embeddings from the GNN
        g_embed = self.gnn(graph, g_embed)
        # -------------version 02--------------
        t_embed = self.embed_layer(traj_inputs[0], traj_inputs[1], traj_inputs[2], traj_inputs[3])
        t_embed = t_embed.unsqueeze(1)  # batch at the second place
        tgt_embed = self.traj_model(t_embed)[-1]

        concern_score = self.out_layer(g_embed).flatten()
        concern_score = F.softmax(concern_score, dim=-1)
        return tgt_embed, g_embed, concern_score


class AttentionCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(AttentionCritic, self).__init__()
        # State embedding
        self.state_embed = nn.Linear(input_dim, hidden_dim)
        # Node embedding
        self.node_embed = nn.Linear(input_dim, hidden_dim)
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        # Output Q-value
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, action_embed, node_embeddings):
        # Embed state (similarity vector a)
        action_embed = self.state_embed(action_embed).unsqueeze(0)
        # Embed node embeddings
        node_emb = self.node_embed(node_embeddings)
        node_emb = node_emb.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)

        # Attention mechanism
        attn_output, _ = self.attention(action_embed, node_emb, node_emb)
        # Q-value output
        q_value = self.q_out(attn_output.squeeze(0)).squeeze(1)
        return q_value


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Model parameters saved to {}".format(path))


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print("Model parameters loaded from {}".format(path))
