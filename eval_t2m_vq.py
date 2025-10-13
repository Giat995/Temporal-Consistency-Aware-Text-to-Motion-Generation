import sys
import os
from os.path import join as pjoin

import torch
from models.vq.model import RVQVAE
from options.vq_option import arg_parse
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
import utils.eval_t2m_tcc as eval_t2m
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from utils.word_vectorizer import WordVectorizer
from utils.fixseed import fixseed
from options.eval_option import EvalT2MOptions
from torch.utils.tensorboard import SummaryWriter
from utils import utils_model
import json

def load_vq_model(vq_opt, device):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm,
                vq_opt.tcc_flag,
                vq_opt.stochastic_matching,
                vq_opt.normalize_embeddings,
                vq_opt.loss_type,
                vq_opt.similarity_type,
                vq_opt.num_cycles,
                vq_opt.cycle_length,
                vq_opt.tcc_loc,
                vq_opt.frb,
                device)
    ckpt = torch.load(pjoin(vq_opt.out_dir, 'net_best_fid.pth'),
                            map_location=device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

if __name__ == "__main__":

    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    eval_dir = pjoin(opt.out_dir, opt.dataset_name, opt.name)
    os.makedirs(eval_dir, exist_ok=True)

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(eval_dir)
    writer = SummaryWriter(eval_dir)
    logger.info(json.dumps(vars(opt), indent=4, sort_keys=True))

    opt.device = torch.device('cuda')
    print(f"Using Device: {opt.device}") 
    torch.autograd.set_detect_anomaly(True)

    vq_opt_path = pjoin(opt.vq_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt, opt.device)


    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
                                                        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    print(len(eval_val_loader))

    vq_model.eval()
    vq_model.to(opt.device)

    

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mae = []
    repeat_time = 20
    for i in range(repeat_time):
        best_fid, best_div, Rprecision, best_matching, l1_dist = \
            eval_t2m.evaluation_vqvae_plus_mpjpe(eval_val_loader, vq_model, i, eval_wrapper=eval_wrapper, num_joint=opt.nb_joints)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        matching.append(best_matching)
        mae.append(l1_dist)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mae = np.array(mae)


    msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n" \
                f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n" \
                f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}\n" \
                f"\tMAE:{np.mean(mae):.3f}, conf.{np.std(mae)*1.96/np.sqrt(repeat_time):.3f}\n\n"
    # logger.info(msg_final)
    print(msg_final)
    logger.info(msg_final)


