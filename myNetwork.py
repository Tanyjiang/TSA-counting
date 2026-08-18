import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import tokenize
from masksembles.torch import Masksembles2D

class SAB1(nn.Module):
    def __init__(self, dim_model, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim_model)
        self.attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim_model)
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * 4, dim_model),
        )
    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x

class SAB2(nn.Module):
    def __init__(self, dim_model, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim_model)
        self.attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim_model)
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * 4, dim_model),
        )
    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x

class network(nn.Module):

    def __init__(self, feature_extractor, flag, num_classes=7):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.feature.requires_grad_(False)
        self.num_classes = num_classes
        self.mask_index = 2
        self.n = 8
        self.reg_layer_mask = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Masksembles2D(512, self.n, 2.0),  # masksembles
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.output_layer_mask = nn.Conv2d(128, num_classes, kernel_size=1)
        nn.init.normal_(self.output_layer_mask.weight, std=0.01)
        nn.init.constant_(self.output_layer_mask.bias, 0)

        self.avp = nn.AdaptiveAvgPool2d((1, 1))
        self.Up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.frontend = nn.Sequential(

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        def init_param(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.frontend.apply(init_param)

        self.output_layer = nn.Conv2d(128, num_classes, kernel_size=1)
        self.relu = nn.ReLU()


        self.fc = nn.Linear(512, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0)

        width = 512
        scale = width ** -0.5
        self.top_k_edges = 3
        self.down_size = 16
        self.down_proj = nn.Linear(width, self.down_size)
        self.up_proj = nn.Linear(self.down_size, width)
        self.edge_proj = nn.Linear(self.down_size * 3, self.down_size)
        self.down_proj2 = nn.Linear(width, self.down_size)
        self.up_proj2 = nn.Linear(self.down_size, width)
        self.edge_proj2 = nn.Linear(self.down_size * 3, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.sab1 = SAB1(self.down_size, num_heads=4, dropout=0.1)
        self.sab2 = SAB2(self.down_size, num_heads=4, dropout=0.1)


    def forward(self, input, text, flag=0, increntmal_phase=0, old=False):
        x, w, x_g = self.feature.encode_image(input, increntmal_phase)

        B, N, D = x_g.shape
        k = self.top_k_edges


        x_g_down = self.non_linear_func(self.down_proj2(x_g))
        x_g_down_norm = F.normalize(x_g_down, dim=-1)

        sim = torch.bmm(x_g_down_norm, x_g_down_norm.transpose(1, 2))
        diag_mask = torch.eye(N, device=x_g.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(diag_mask, float("-1e9"))

        topk_vals, topk_idx = torch.topk(sim, k=k, dim=-1, largest=False, sorted=False)

        x_g_i = x_g_down.unsqueeze(2).expand(-1, -1, k, -1)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, x_g_down.shape[-1])
        x_g_j = torch.gather(x_g_down.unsqueeze(1).expand(-1, N, -1, -1), 2, topk_idx_exp)
        diff = x_g_i - x_g_j
        edge_feat = torch.cat([x_g_i, x_g_j, diff], dim=-1)
        edge_feat = edge_feat.view(B, N * k, -1)

        edge_feat = self.non_linear_func(self.edge_proj2(edge_feat))
        edge_feat = self.sab2(edge_feat)

        device = x.device
        src_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, k).reshape(-1)
        src_idx_batched = src_idx.unsqueeze(0).expand(B, -1)
        out = torch.zeros(B, N, self.down_size, device=device, dtype=edge_feat.dtype)
        index_for_scatter = src_idx_batched.unsqueeze(-1).expand(-1, -1, self.down_size)
        out.scatter_add_(1, index_for_scatter, edge_feat)
        out = out / float(k)

        out_final = self.up_proj2(out)
        out_final_to_text = out_final

        if text is not None:
            if flag == 1:
                y = self.avp(x)
                y = y.view(y.size(0), -1)
                y = self.fc(y)
                y = self.sigmoid(y)
                y = torch.argmax(y)
                t = text[0][str(y.cpu().numpy())]
                text_all = []
                for i in range(len(t)):
                    list = tokenize(t[i])
                    text_all.append(list)
                text = [text_all]

            t_cat = []
            for text_i in text:
                text_i_cat = torch.cat(text_i, dim=0)
                t, w_t = self.feature.encode_text(text_i_cat.to('cuda'), increntmal_phase)
                t_cat.append(t)
            t_cat = torch.stack(t_cat)

            B, N, D = out_final_to_text.shape
            ref = out_final_to_text.mean(dim=1)  # [B, D]

            output_norm = F.normalize(out_final_to_text, dim=-1)  # [B, N, D]
            ref_norm = F.normalize(ref, dim=-1).unsqueeze(1)  # [B, 1, D]

            sim = (output_norm * ref_norm).sum(dim=-1)

            K = 10
            _, idx = torch.topk(sim, k=K, dim=-1)

            out_final_to_text = torch.gather(
                out_final_to_text,
                dim=1,
                index=idx.unsqueeze(-1).expand(-1, -1, D)
            )
            t_cat = torch.cat([out_final_to_text, t_cat], dim=1)


            B, N, D = t_cat.shape
            k = self.top_k_edges


            t_cat_down = self.non_linear_func(self.down_proj(t_cat))
            t_cat_down_norm = F.normalize(t_cat_down, dim=-1)

            sim_t = torch.bmm(t_cat_down_norm, t_cat_down_norm.transpose(1, 2))
            diag_mask_t = torch.eye(N, device=t_cat.device, dtype=torch.bool).unsqueeze(0)
            sim_t = sim_t.masked_fill(diag_mask_t, float("-1e9"))

            topk_vals_t, topk_idx_t = torch.topk(sim_t, k=k, dim=-1, largest=False, sorted=False)

            t_cat_i = t_cat_down.unsqueeze(2).expand(-1, -1, k, -1)
            topk_idx_exp_t = topk_idx_t.unsqueeze(-1).expand(-1, -1, -1, t_cat_down.shape[-1])
            t_cat_j = torch.gather(t_cat_down.unsqueeze(1).expand(-1, N, -1, -1), 2, topk_idx_exp_t)
            diff_t = t_cat_i - t_cat_j
            edge_feat_t = torch.cat([t_cat_i, t_cat_j, diff_t], dim=-1)
            edge_feat_t = edge_feat_t.view(B, N * k, -1)

            edge_feat_t = self.non_linear_func(self.edge_proj(edge_feat_t))
            edge_feat_t = self.sab1(edge_feat_t)

            src_idx_t = torch.arange(N, device=x.device).unsqueeze(1).expand(N, k).reshape(-1)
            src_idx_batched_t = src_idx_t.unsqueeze(0).expand(B, -1)
            out_t = torch.zeros(B, N, self.down_size, device=x.device, dtype=edge_feat_t.dtype)
            index_for_scatter_t = src_idx_batched_t.unsqueeze(-1).expand(-1, -1, self.down_size)
            out_t.scatter_add_(1, index_for_scatter_t, edge_feat_t)
            out_t = out_t / float(k)

            out_final_t = self.non_linear_func(self.up_proj(out_t).mean(dim=1))


        out_final = out_final.reshape(B, 14, 14, -1).permute(0, 3, 1, 2).contiguous()

        if text is not None:
            out_final = out_final_t.unsqueeze(2).unsqueeze(2) * out_final + x
        else:
            out_final = out_final + x


        x = self.Up2(out_final)

        if old:
            x_mask = None
        else:
            if self.training:
                x_mask = self.reg_layer_mask(x)
            else:
                x_mask = x
                for index in range(len(self.reg_layer_mask)):
                    if index == self.mask_index:
                        x_mask = x_mask.repeat_interleave(self.n, 0)
                        x_mask = torch.split(x_mask.unsqueeze(1), 1, dim=0)
                        x_mask = torch.cat(x_mask, dim=1).permute([1, 0, 2, 3, 4])
                        x_mask = x_mask * self.reg_layer_mask[index].masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                        x_mask = torch.cat(torch.split(x_mask, 1, dim=0),
                                           dim=1)
                        x_mask = x_mask.squeeze(0).float()
                    else:
                        x_mask = self.reg_layer_mask[index](x_mask)
                x_mask = x_mask.mean(0).unsqueeze(0)
            x_mask = F.interpolate(x_mask, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        if not old:
            x_mask = self.output_layer_mask(x_mask)

        y = self.avp(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        y = self.sigmoid(y)

        x = self.frontend(x)

        z = self.avp(x)
        z = z.view(z.size(0), -1)

        x = self.output_layer(x)
        x = self.relu(x)

        if flag == 1:
            x = (F.interpolate(x, size=(x.size(2) * 8, x.size(3) * 8), mode="bilinear", align_corners=True, ) / 64)

        return x, y, z, x_mask, w

    def Incremental_learning_weight(self, numclass):
        data = self.output_layer.weight
        bias = self.output_layer.bias
        old_num = self.output_layer.out_channels
        self.output_layer = nn.Conv2d(128, out_channels=numclass + 1, kernel_size=1)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            with torch.no_grad():
                nn.init.constant_(self.output_layer.bias, 0)
        with torch.no_grad():
            self.output_layer.weight[:old_num] = nn.Parameter(data)
            self.output_layer.bias[:old_num] = nn.Parameter(bias)

        weight_fc = self.fc.weight.data
        bias_fc = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features
        self.fc = nn.Linear(in_feature, numclass + 1, bias=True)
        self.fc.weight.data[:out_feature] = weight_fc
        self.fc.bias.data[:out_feature] = bias_fc

        # mask branch
        data = self.output_layer_mask.weight
        bias = self.output_layer_mask.bias
        old_num = self.output_layer_mask.out_channels
        self.output_layer_mask = nn.Conv2d(128, out_channels=numclass + 1, kernel_size=1)
        nn.init.normal_(self.output_layer_mask.weight, std=0.01)
        if self.output_layer_mask.bias is not None:
            with torch.no_grad():
                nn.init.constant_(self.output_layer_mask.bias, 0)
        with torch.no_grad():
            self.output_layer_mask.weight[:old_num] = nn.Parameter(data)
            self.output_layer_mask.bias[:old_num] = nn.Parameter(bias)

    def Incremental_learning_head(self, numclass):
        pass

    def feature_extractor(self, inputs):
        return self.feature(inputs)
