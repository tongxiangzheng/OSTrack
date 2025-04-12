"""
Basic OSTrack model.
"""
import math
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.layers.neck import build_neck
from lib.models.ostrack import fastitpn as fastitpn_module
from lib.models.layers.encoder import build_encoder

class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, encoder, neck, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.encoder = encoder
        self.box_head = box_head
        self.neck = neck

        self.aux_loss = aux_loss
        self.head_type = head_type
        self.num_patch_x = self.encoder.body.num_patches_search
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template = None,
                search = None,
                template_anno_list=None,
                enc_opt=None,
                neck_h_state=None,
                feature=None,
                mode='encoder'
                ):
        if mode == "encoder":
            return self.forward_encoder(template, search, template_anno_list)
        elif mode == "neck":
            return self.forward_neck(enc_opt,neck_h_state)
        elif mode == "decoder":
            return self.forward_decoder(feature)
        else:
            raise ValueError
    
    def forward_neck(self, enc_out, neck_h_state):
        x = enc_out
        num_patch_x = self.num_patch_x
        xs = x[:, 0:num_patch_x]
        interaction_indexes=[[8, 14], [14, 20], [20, 26],[26,32]]
        x,xs,h = self.neck(x,xs,neck_h_state,self.encoder.body.blocks,interaction_indexes)
        x = self.encoder.body.fc_norm(x)
        xs = xs + x[:, 0:num_patch_x]
        return x,xs,h
    
    def forward_decoder(self, feature):
        return self.forward_head(feature, None)

    def forward_encoder(self, template, search, template_anno_list):
        x = self.encoder(template,search,template_anno_list)
        return x

    def forward_head(self, feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        #enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        #opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, HW, C = feature.size()
        feature = feature.permute((0, 2, 1)).contiguous()
        feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            
            score_map_ctr, bbox, size_map, offset_map = self.box_head(feature, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    # if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
    #     backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1

    # elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
    #     backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                        )
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1

    # elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
    #     backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                         ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                         ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                         )

    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    # elif cfg.MODEL.BACKBONE.TYPE == 'fastitpnb':
    #     backbone = fastitpn_module.fastitpnb(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                         ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                         ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                         )

    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1

    # else:
    #     raise NotImplementedError

    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)



    encoder = build_encoder(cfg)
    neck = build_neck(cfg,encoder)
    box_head = build_box_head(cfg, encoder.num_channels)

    model = OSTrack(
        encoder,
        neck,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
