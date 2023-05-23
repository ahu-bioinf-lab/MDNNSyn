import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, SAGEConv, global_max_pool, global_mean_pool
import sys
sys.path.append('..')
from utils import reset

drug_num = 38
cline_num = 32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HgnnEncoder, self).__init__()
        self.ln = torch.nn.Linear(in_channels, 256, bias=False)

        self.conv1 = HypergraphConv(in_channels, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.convs = nn.ModuleList([HypergraphConv(256, 256) for i in range (3)] )
        self.batchs = nn.ModuleList([nn.BatchNorm1d(256) for i in range (3)])
        self.fc1s = nn.ModuleList([torch.nn.Linear(256, 128) for i in range (3)])
        self.fc2s = nn.ModuleList([torch.nn.Linear(128, 256) for i in range (3)])

        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.sig = nn.Sigmoid()
        self.act = nn.ReLU()
        self.drop_out = nn.Dropout(0.4)
        self.conv3 = HypergraphConv(256, out_channels)

    def forward(self, x, edge):
        # 残差块
        x0 = self.ln(x)
        p = 0.2
        edge_mask = torch.bernoulli(torch.ones_like(edge)*(1-p)).bool()
        edge = edge * edge_mask

        x1 = self.act(self.conv1(x, edge))
        s1 = self.act(self.fc1(x1))
        s1 = F.avg_pool1d(s1.unsqueeze(0),kernel_size=1,stride=1)
        s1 = s1.squeeze(0)
        s1 = self.sig(self.fc2(s1))
        x = self.drop_out(self.batch1(x1 + s1 + x0))

        for i in range(3):
            x = self.act(self.convs[i](x, edge))
            x = self.drop_out(self.batchs[i](0.8 * x + 0.2 * x0))
            s = self.act(self.fc1s[i](x))
            s = F.avg_pool1d(s.unsqueeze(0), kernel_size=1, stride=1)
            s = s.squeeze(0)
            x = self.sig(self.fc2s[i](s))

        x = self.act(self.conv3(x, edge))
        return x


class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super(BioEncoder, self).__init__()
        # -------drug_layer
        self.use_GMP = use_GMP
        self.conv1 = SAGEConv(dim_drug, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = SAGEConv(128, output)
        self.batch_conv2 = nn.BatchNorm1d(output)
        # -------cell line_layer
        self.conv_cell1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=16)  # 651*8
        self.fc_cell11 = nn.Linear(3 * 636, 100)
        self.batch_cell1 = nn.BatchNorm1d(100)
        self.fc_cell = nn.Linear(1 * 113, output)
        self.reset_para()
        self.act = nn.ReLU()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data):
        # -----drug_train
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.act(x_drug)
        x_drug = self.batch_conv2(x_drug)
        if self.use_GMP:
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)

        # ----cellline_train
        embedded_cell = gexpr_data.unsqueeze(1)
        conv_cell = self.conv_cell1(embedded_cell)
        x_cellline = conv_cell.view(-1, 3 * 636)
        x_cellline = self.fc_cell11(x_cellline)
        x_cellline = self.batch_cell1(x_cellline)

        '''s_drug = x_drug.detach().cpu().numpy()
        drug_sim_matrix = np.array(get_Jaccard_Similarity(s_drug))
        drug_sim_fea = torch.tensor(drug_sim_matrix).cuda()
        s_cellline = x_cellline.detach().cpu().numpy()
        cline_sim_matrix = np.array(get_Jaccard_Similarity(s_cellline))
        cline_sim_fea = torch.tensor(cline_sim_matrix).cuda()'''

        return x_drug, x_cellline


class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()


        self.d_fc = nn.Linear(294, 256)
        self.c_fc = nn.Linear(288, 256)

        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.att1 = nn.Linear(in_channels // 2, in_channels // 2)
        self.att2 = nn.Linear(in_channels // 2, 1, bias=False)

        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)

        self.reset_parameters()
        self.drop_out = nn.Dropout(0.5)
        self.act = nn.Tanh()
        self.trans = nn.Linear(512, 256)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, drug_sim_mat, cline_sim_mat, druga_id, drugb_id, cellline_id):
        drug_emb, cline_emb = graph_embed[:drug_num], graph_embed[drug_num:]
        drug_fea = torch.cat((drug_emb, drug_sim_mat), 1)
        cline_fea = torch.cat((cline_emb, cline_sim_mat), 1).to(torch.float)

        drug_fea = self.d_fc(drug_fea)
        cline_fea = self.c_fc(cline_fea)
        graph_embed = torch.cat((drug_fea, cline_fea), 0)
        h0 = torch.cat((graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)
        #h0 = self.cross(h0)
        h = self.act(self.fc1(h0))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h2 = self.fc3(h)

        item_regularizer = (torch.norm(graph_embed[druga_id, :]) ** 2
                            + torch.norm(graph_embed[drugb_id, :]) ** 2
                            + torch.norm(graph_embed[cellline_id, :]) ** 2) / 2
        emb_loss = 1e-3 * item_regularizer / graph_embed[cellline_id, :].shape[0]
        return torch.sigmoid(h2.squeeze(dim=1)), emb_loss, h

    def cross(self, x):
        m, n = x.shape
        layer1 = torch.nn.Linear(n, n).cuda()
        layer2 = torch.nn.Linear(m, m).cuda()
        x1 = layer1(x)
        x2 = layer2(x1.permute(1,0)).permute(1,0).cuda()
        scale = x.size(1) ** -0.5
        attention = F.softmax(x * scale, dim=1)
        res = attention * x2
        return x + res


class HypergraphSynergy(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder):
        super(HypergraphSynergy, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.bio_encoder)
        reset(self.graph_encoder)
        reset(self.decoder)

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, adj, drug_sim_mat, cline_sim_mat, druga_id, drugb_id, cellline_id):
        drug_embed, cellline_embed = self.bio_encoder(drug_feature, drug_adj, ibatch, gexpr_data)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.graph_encoder(merge_embed, adj)
        res, emb_loss, h = self.decoder(graph_embed, drug_sim_mat, cline_sim_mat, druga_id, drugb_id, cellline_id)
        return res, emb_loss, h
