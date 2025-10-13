import random
import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
from models.vq.residual_vq import ResidualVQ
from models.vq.tcc.alignment import compute_alignment_loss
from common.skeleton import do_smplxfk, SMPLX_Skeleton, ax_from_6v, recover_from_ric

class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond):
        for layer in self.stack:
            x = layer(x, cond)
        return x
    
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 tcc_flag=False,
                 stochastic_matching=False,
                 normalize_embeddings=False, 
                 loss_type='regression_mse',
                 similarity_type='l2',
                 num_cycles=20,
                 cycle_length=2,
                 tcc_loc='encoder',
                 frb_flag=False,
                 device='cpu'):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        self.tcc_flag = tcc_flag
        self.stochastic_matching = stochastic_matching
        self.normalize_embeddings = normalize_embeddings
        self.loss_type = loss_type
        self.similarity_type = similarity_type
        self.num_cycles = num_cycles
        self.cycle_length = cycle_length
        self.tcc_loc = tcc_loc
        self.frb_flag=frb_flag
        if input_width==263:
            self.num_joints=22
        elif input_width==251:
            self.num_joints=21
        
        self.refine_input_projection = nn.Linear(input_width, code_dim)
        self.refine_cond_projection = nn.Linear(48, code_dim)
        #self.smplx_fk = SMPLX_Skeleton(Jpath='common/smplx_neu_J_1.npy', device=device)
        decoderlayer = nn.TransformerDecoderLayer(code_dim, nhead=8)
        self.refine_seqTransDecoder = nn.TransformerDecoder(decoder_layer=decoderlayer, num_layers=6)
        self.refine_final_layer = nn.Linear(code_dim, input_width)
        #self.refine_final_layer2 = nn.Linear(263, 263)
        



    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes
    
    def get_rcond(self, output):
        # with torch.no_grad():
            #joints3d = do_smplxfk(output, self.smplx_fk)[:,:,:22,:]
            joints3d = recover_from_ric(output, self.num_joints)
            B,T,J,_ = joints3d.shape
            if self.num_joints==22:
                l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
            elif self.num_joints==21:
                l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 19, 14, 20, 15
            relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
            pred_foot = joints3d[:, :, relevant_joints, :]          # B,T,J,4
            foot_vel = torch.zeros_like(pred_foot)
            foot_vel[:, :-1] = (
                pred_foot[:, 1:, :, :] - pred_foot[:, :-1, :, :]
            )  # (N, S-1, 4, 3)
            foot_y_ankle = pred_foot[:, :, :2, 1]
            foot_y_toe = pred_foot[:, :, 2:, 1]
            fc_mask_ankle = torch.unsqueeze((foot_y_ankle <= (-1.2+0.012)), dim=3).repeat(1, 1, 1, 3)
            fc_mask_teo = torch.unsqueeze((foot_y_toe <= (-1.2+0.05)), dim=3).repeat(1, 1, 1, 3)
            contact_lable = torch.cat([fc_mask_ankle, fc_mask_teo], dim=2).int().to(output).reshape(B, T, -1)

            contact_toe_thresh, contact_ankle_thresh, contact_vel_thresh = -1.2+0.08, -1.2+0.015, 0.3 / 30           # 30 is fps
            contact_score_ankle = torch.sigmoid((contact_ankle_thresh - pred_foot[:, :, :2, 1])/contact_ankle_thresh*5) * torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, :2, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_ankle = torch.unsqueeze(contact_score_ankle, dim=3).repeat(1, 1, 1, 3)
            contact_score_toe = torch.sigmoid((contact_toe_thresh - pred_foot[:, :, 2:, 1])/contact_toe_thresh*5) * torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, 2:, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_toe = torch.unsqueeze(contact_score_toe, dim=3).repeat(1, 1, 1, 3)
            contact_score = torch.cat([contact_score_ankle, contact_score_toe], dim = -2).reshape(B, T, -1)
            r_cond = torch.cat([contact_lable, contact_score, pred_foot.reshape(B,T,-1), foot_vel.reshape(B,T,-1)], dim = -1) 
            return r_cond

    def forward(self, x):

        b = x.shape[0]

        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)

        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)

        if self.frb_flag:
            r_output = self.refine_input_projection(x_out) #32*196*512
            r_cond = self.get_rcond(x_out).to(r_output.dtype) #32*196*48
            r_cond = self.refine_cond_projection(r_cond) #32*196*512
            refine_output = self.refine_seqTransDecoder(r_output, r_cond)
            refine_output = self.refine_final_layer(refine_output)
            x_out = refine_output
            
            '''
            joint_out = x_out[..., :139] #32*196*139s
            r_output = self.refine_input_projection(joint_out) #32*196*512
            r_cond = self.get_rcond(joint_out, joint) #32*196*48
            r_cond = self.refine_cond_projection(r_cond) #32*196*512
            refine_output = self.refine_seqTransDecoder(r_output, r_cond)
            refine_output = self.refine_final_layer(refine_output)
            out = torch.zeros_like(x_out)
            out[..., :139] = x_out[..., :139] + refine_output
            out[..., 139:] = x_out[..., 139:]
            if self.projection_flag:
                out = self.refine_final_layer2(out)
            '''
        
        
        if self.tcc_flag:
            if self.tcc_loc == 'encoder':
                tcc_loss = compute_alignment_loss(x_encoder, b, None, None, self.stochastic_matching, self.normalize_embeddings, self.loss_type, self.similarity_type, self.num_cycles, self.cycle_length)
            else:
                tcc_loss = compute_alignment_loss(x_quantized, b, None, None, self.stochastic_matching, self.normalize_embeddings, self.loss_type, self.similarity_type, self.num_cycles, self.cycle_length)
            return x_out, commit_loss, perplexity, tcc_loss
            

        # x_out = self.postprocess(x_decoder)

        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        if self.frb_flag:
            # joint_out = x_out[..., :139] #32*196*139
            # r_output = self.refine_input_projection(joint_out) #32*196*512
            # r_cond = self.get_rcond(joint_out) #32*196*48
            # r_cond = self.refine_cond_projection(r_cond) #32*196*512
            # refine_output = self.refine_seqTransDecoder(r_output, r_cond)
            # refine_output = self.refine_final_layer(refine_output)
            # out = torch.zeros_like(x_out)
            # out[..., :139] = x_out[..., :139] + refine_output
            # out[..., 139:] = x_out[..., 139:]
            # if self.projection_flag:
            #     out = self.refine_final_layer2(out)
            r_output = self.refine_input_projection(x_out) #32*196*512
            r_cond = self.get_rcond(x_out).to(r_output.dtype) #32*196*48
            r_cond = self.refine_cond_projection(r_cond) #32*196*512
            refine_output = self.refine_seqTransDecoder(r_output, r_cond)
            out = self.refine_final_layer(refine_output)
            return out
        else:
            return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)
    
class FootRefineModel(nn.Module):
    def __init__(self,
                 dim_pose=263,
                 code_dim=512,
                 device='cpu'):

        super().__init__()
        self.code_dim = code_dim
        self.dim_pose = dim_pose

        self.refine_input_projection = nn.Linear(139, code_dim)
        self.refine_cond_projection = nn.Linear(48, code_dim)
        self.smplx_fk = SMPLX_Skeleton(Jpath='common/smplx_neu_J_1.npy', device=device)
        decoderlayer = nn.TransformerDecoderLayer(code_dim, nhead=8)
        self.refine_seqTransDecoder = nn.TransformerDecoder(decoder_layer=decoderlayer, num_layers=6)
        self.refine_final_layer1 = nn.Linear(code_dim, 139)
        self.refine_final_layer2 = nn.Linear(dim_pose, dim_pose)

    
    def get_rcond(self, output):
        # with torch.no_grad():
            joints3d = do_smplxfk(output, self.smplx_fk)[:,:,:22,:]
            B,T,J,_ = joints3d.shape
            l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
            relevant_joints = [l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx]
            pred_foot = joints3d[:, :, relevant_joints, :]          # B,T,J,4
            foot_vel = torch.zeros_like(pred_foot)
            foot_vel[:, :-1] = (
                pred_foot[:, 1:, :, :] - pred_foot[:, :-1, :, :]
            )  # (N, S-1, 4, 3)
            foot_y_ankle = pred_foot[:, :, :2, 1]
            foot_y_toe = pred_foot[:, :, 2:, 1]
            fc_mask_ankle = torch.unsqueeze((foot_y_ankle <= (-1.2+0.012)), dim=3).repeat(1, 1, 1, 3)
            fc_mask_teo = torch.unsqueeze((foot_y_toe <= (-1.2+0.05)), dim=3).repeat(1, 1, 1, 3)
            contact_lable = torch.cat([fc_mask_ankle, fc_mask_teo], dim=2).int().to(output).reshape(B, T, -1)

            contact_toe_thresh, contact_ankle_thresh, contact_vel_thresh = -1.2+0.08, -1.2+0.015, 0.3 / 30           # 30 is fps
            contact_score_toe = torch.sigmoid((contact_toe_thresh - pred_foot[:, :, :2, 1])/contact_toe_thresh*5) * \
            torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, :2, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_toe = torch.unsqueeze(contact_score_toe, dim=3).repeat(1, 1, 1, 3)
            contact_score_ankle = torch.sigmoid((contact_ankle_thresh - pred_foot[:, :, 2:, 1])/contact_ankle_thresh*5) * \
            torch.sigmoid((contact_vel_thresh - torch.norm(foot_vel[:, :, 2:, [0, 2]], dim=-1))/contact_vel_thresh*5)
            contact_score_ankle = torch.unsqueeze(contact_score_ankle, dim=3).repeat(1, 1, 1, 3)
            contact_score = torch.cat([contact_score_ankle, contact_score_ankle], dim = -2).reshape(B, T, -1)
            r_cond = torch.cat([contact_lable, contact_score, pred_foot.reshape(B,T,-1), foot_vel.reshape(B,T,-1)], dim = -1) 
            return r_cond

    def forward(self, x_out):


        joint_out = x_out[..., :139] #32*196*139
        r_output = self.refine_input_projection(joint_out) #32*196*512
        r_cond = self.get_rcond(joint_out) #32*196*48
        r_cond = self.refine_cond_projection(r_cond) #32*196*512
        refine_output = self.refine_seqTransDecoder(r_output, r_cond)
        refine_output = self.refine_final_layer1(refine_output)
        out = torch.zeros_like(x_out)
        out[..., :139] = x_out[..., :139] + refine_output
        out[..., 139:] = x_out[..., 139:]
        #out = self.refine_final_layer2(out)
    
        return out

class RVQVAE_raw(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 tcc_flag=False,
                 stochastic_matching=False,
                 normalize_embeddings=False, 
                 loss_type='regression_mse',
                 similarity_type='l2',
                 num_cycles=20,
                 cycle_length=2,
                 tcc_loc='encoder',
                 device='cpu'):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        self.tcc_flag = tcc_flag
        self.stochastic_matching = stochastic_matching
        self.normalize_embeddings = normalize_embeddings
        self.loss_type = loss_type
        self.similarity_type = similarity_type
        self.num_cycles = num_cycles
        self.cycle_length = cycle_length
        self.tcc_loc = tcc_loc
        
        



    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes
    

    def forward(self, x):

        b = x.shape[0]

        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)

        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)

        
        
        if self.tcc_flag:
            if self.tcc_loc == 'encoder':
                tcc_loss = compute_alignment_loss(x_encoder, b, None, None, self.stochastic_matching, self.normalize_embeddings, self.loss_type, self.similarity_type, self.num_cycles, self.cycle_length)
            else:
                tcc_loss = compute_alignment_loss(x_quantized, b, None, None, self.stochastic_matching, self.normalize_embeddings, self.loss_type, self.similarity_type, self.num_cycles, self.cycle_length)
            return x_out, commit_loss, perplexity, tcc_loss
            

        # x_out = self.postprocess(x_decoder)

        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        
        return x_out
    