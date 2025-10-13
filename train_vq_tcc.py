
import os
import json
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vq.model import RVQVAE
from models.vq.vq_trainer_tcc import RVQTokenizerTrainer
from options.vq_option import arg_parse
from data.t2m_dataset import MotionDataset, cycle
from data import tcc_vq_dataset
from utils import paramUtil
import numpy as np
import utils.utils_model as utils_model
from exit.utils import init_save_folder

from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.fixseed import fixseed
from torch.utils.tensorboard import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"

def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)
    fixseed(opt.seed)

    opt.out_dir = os.path.join(opt.out_dir, f'vq', opt.dataset_name)

    init_save_folder(opt)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.out_dir, 'model')
    opt.meta_dir = pjoin(opt.out_dir, 'meta')
    opt.eval_dir = pjoin(opt.out_dir, 'animation')
    
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.out_dir, exist_ok=True)

    if opt.is_train:
    # save to the disk
        args = vars(opt)
        file_name = os.path.join(opt.out_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(opt.out_dir)
    writer = SummaryWriter(opt.out_dir)
    logger.info(json.dumps(vars(opt), indent=4, sort_keys=True))


    opt.device = torch.device('cuda')
    print(f"Using Device: {opt.device}")   

    if opt.dataset_name == "t2m":
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joint_dir = pjoin(opt.data_root, 'new_joints')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        dim_pose = 263
        fps = 20
        radius = 4
        kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == "kit":
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joint_dir = pjoin(opt.data_root, 'new_joints')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
    else:
        raise KeyError('Dataset Does not Exists')

    logger.info(f'Training on {opt.dataset_name}, motions are with {opt.joints_num} joints')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')


    net = RVQVAE(opt,
                dim_pose,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm,
                opt.tcc_flag,
                opt.stochastic_matching,
                opt.normalize_embeddings,
                opt.loss_type,
                opt.similarity_type,
                opt.num_cycles,
                opt.cycle_length,
                opt.tcc_loc,
                opt.frb,
                opt.device)

    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)
    # print("Total parameters of discriminator net: {}".format(pc_vq))
    # all_params += pc_vq_dis

    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))

    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    
    '''train_dataset = MotionDataset(opt, mean, std, train_split_file)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    train_loader_iter = cycle(train_loader)'''
    
                            
    train_loader = tcc_vq_dataset.DATALoader(opt, opt.dataset_name,
                                        opt.batch_size,
                                        window_size=opt.window_size,
                                        unit_length=2**opt.down_t, mode='train')

    train_loader_iter = tcc_vq_dataset.cycle(train_loader)
    
    val_dataset = MotionDataset(opt, mean, std, val_split_file)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)
    trainer.train(train_loader_iter, val_loader, eval_val_loader, eval_wrapper, logger, writer, plot_t2m)

## train_vq.py --dataset_name kit --batch_size 512 --name VQVAE_dp2 --gpu_id 3
## train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp2_b256 --gpu_id 2
## train_vq.py --dataset_name kit --batch_size 1024 --name VQVAE_dp2_b1024 --gpu_id 1
## python train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp1_b256 --gpu_id 2