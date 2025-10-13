import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn.functional as F

import torch.optim as optim

import time
import math
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from utils.eval_t2m_tcc import evaluation_vqvae, evaluation_frb
from utils.utils import print_current_loss

import os
import sys

def def_value():
    return 0.0


class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        if args.is_train:
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()

        # self.critic = CriticWrapper(self.opt.dataset_name, self.opt.device)


    def forward(self, batch_data):
        #batch_data = batch_data[0]
        motions = batch_data.detach().to(self.device).float()
        if self.vq_model.tcc_flag:
            pred_motion, loss_commit, perplexity, loss_tcc = self.vq_model(motions)
        else:
            pred_motion, loss_commit, perplexity = self.vq_model(motions)
        
        self.motions = motions
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(pred_motion, motions)
        pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)
        if self.vq_model.tcc_flag:
            loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit + self.opt.loss_tcc * loss_tcc
            return loss, loss_rec, loss_explicit, loss_commit, perplexity, loss_tcc
        else:
            loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit
            return loss, loss_rec, loss_explicit, loss_commit, perplexity

        # return loss, loss_rec, loss_vel, loss_commit, perplexity
        # return loss, loss_rec, loss_percept, loss_commit, perplexity
        


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader_iter, val_loader, eval_val_loader, eval_wrapper, logger, writer, plot_eval=None):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            logger.info("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_iter
        # val_loss = 0
        # min_val_loss = np.inf
        # min_val_epoch = epoch
        current_lr = self.opt.lr

        # sys.exit()
        best_iter, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger, writer = evaluation_vqvae(
            self.opt.out_dir, eval_val_loader, self.vq_model, logger, writer, it=0, best_iter=0, best_fid=1000,
            best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper, save=False)

        avg_loss, avg_recons, avg_vel, avg_perplexity, avg_commit, avg_tcc, lr = 0., 0., 0., 0., 0., 0., 0.

        
        for it in tqdm(range(1, self.opt.max_iter + 1)):
            self.vq_model.train()
            sample = next(train_loader_iter)
            batch_data = sample[0]
            #joint = sample[1]
            
            if it < self.opt.warm_up_iter:
                current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
            if self.opt.tcc_flag:
                loss, loss_rec, loss_vel, loss_commit, perplexity, loss_tcc = self.forward(batch_data)
            else:
                loss, loss_rec, loss_vel, loss_commit, perplexity = self.forward(batch_data)
            self.opt_vq_model.zero_grad()
            loss.backward()
            clip_grad_value_(self.vq_model.parameters(), clip_value=0.1)
            self.opt_vq_model.step()

            if it >= self.opt.warm_up_iter:
                self.scheduler.step()
            
            avg_loss += loss.item()
            avg_recons += loss_rec.item()
            # Note it not necessarily velocity, too lazy to change the name now
            avg_vel += loss_vel.item()
            avg_commit += loss_commit.item()
            avg_perplexity += perplexity.item()
            if self.opt.tcc_flag:
                avg_tcc += loss_tcc.item()

            if it % self.opt.print_iter ==  0 :
                avg_loss /= self.opt.print_iter
                avg_recons /= self.opt.print_iter
                avg_vel /= self.opt.print_iter
                avg_commit /= self.opt.print_iter
                avg_perplexity /= self.opt.print_iter
                writer.add_scalar('./Train/L1', avg_recons, it)
                writer.add_scalar('./Train/PPL', avg_perplexity, it)
                writer.add_scalar('./Train/Commit', avg_commit, it)
                writer.add_scalar('./Train/Vel', avg_vel, it)
                if self.opt.tcc_flag:
                    avg_tcc /= self.opt.print_iter
                    writer.add_scalar('./Train/Tcc', avg_tcc, it)

                if self.opt.tcc_flag:
                    message = f"Iter:{it:6d}\t lr:{self.opt_vq_model.param_groups[0]['lr']:.6f}\t loss:{avg_loss:.5f}\t loss_rec:{avg_recons:.5f}\t loss_vel:{avg_vel:.5f}\t loss_commit:{avg_commit:.5f}\t perplexity:{avg_perplexity:.5f}\t loss_tcc:{avg_tcc:.5f}"
                else:
                    message = f"Iter:{it:6d}\t lr:{self.opt_vq_model.param_groups[0]['lr']:.6f}\t loss:{avg_loss:.5f}\t loss_rec:{avg_recons:.5f}\t loss_vel:{avg_vel:.5f}\t loss_commit:{avg_commit:.5f}\t perplexity:{avg_perplexity:.5f}"

                logger.info(message)

                avg_loss, avg_recons, avg_vel, avg_perplexity, avg_commit, avg_tcc, lr = 0., 0., 0., 0., 0., 0., 0.

            '''if it % self.opt.save_latest == 0:
                torch.save()
                self.save(pjoin(self.opt.out_dir, 'latest.tar'), epoch, it)'''

            if it % self.opt.eval_iter ==  0 :
                best_iter, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger, writer = evaluation_vqvae(
                self.opt.out_dir, eval_val_loader, self.vq_model, logger, writer, it, best_iter=best_iter, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1,
                best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)

            if it == self.opt.max_iter :
                msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}, Matching. {best_matching:.4f}"
                logger.info(msg_final)


            


class LengthEstTrainer(object):

    def __init__(self, args, estimator, text_encoder, encode_fnc):
        self.opt = args
        self.estimator = estimator
        self.text_encoder = text_encoder
        self.encode_fnc = encode_fnc
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            self.logger = SummaryWriter(args.log_dir)
            self.mul_cls_criterion = torch.nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        # self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            # 'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def train(self, train_dataloader, val_dataloader):
        self.estimator.to(self.device)
        self.text_encoder.to(self.device)

        self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        logs = defaultdict(float)
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.estimator.train()

                conds, _, m_lens = batch_data
                # word_emb = word_emb.detach().to(self.device).float()
                # pos_ohot = pos_ohot.detach().to(self.device).float()
                # m_lens = m_lens.to(self.device).long()
                text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device).detach()
                # print(text_embs.shape, text_embs.device)

                pred_dis = self.estimator(text_embs)

                self.zero_grad([self.opt_estimator])

                gt_labels = m_lens // self.opt.unit_length
                gt_labels = gt_labels.long().to(self.device)
                # print(gt_labels.shape, pred_dis.shape)
                # print(gt_labels.max(), gt_labels.min())
                # print(pred_dis)
                acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)
                loss = self.mul_cls_criterion(pred_dis, gt_labels)

                loss.backward()

                self.clip_norm([self.estimator])
                self.step([self.opt_estimator])

                logs['loss'] += loss.item()
                logs['acc'] += acc.item()

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    # self.logger.add_scalar('Val/loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.add_scalar("Train/%s"%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(float)
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            print('Validation time:')

            val_loss = 0
            val_acc = 0
            # self.estimator.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.estimator.eval()

                    conds, _, m_lens = batch_data
                    # word_emb = word_emb.detach().to(self.device).float()
                    # pos_ohot = pos_ohot.detach().to(self.device).float()
                    # m_lens = m_lens.to(self.device).long()
                    text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device)
                    pred_dis = self.estimator(text_embs)

                    gt_labels = m_lens // self.opt.unit_length
                    gt_labels = gt_labels.long().to(self.device)
                    loss = self.mul_cls_criterion(pred_dis, gt_labels)
                    acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)

                    val_loss += loss.item()
                    val_acc += acc.item()


            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            print('Validation Loss: %.5f Validation Acc: %.5f' % (val_loss, val_acc))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss


class FootRefineTrainer:
    def __init__(self, args, foot_refine_model, vq_model):
        self.opt = args
        self.foot_refine_model = foot_refine_model
        self.vq_model = vq_model
        self.device = args.device

        if args.recons_loss == 'l1':
            self.l1_criterion = torch.nn.L1Loss()
        elif args.recons_loss == 'l1_smooth':
            self.l1_criterion = torch.nn.SmoothL1Loss()

        # self.critic = CriticWrapper(self.opt.dataset_name, self.opt.device)


    def forward(self, batch_data):
        #batch_data = batch_data[0]
        conds, motions, m_lens = batch_data
        motions = motions.detach().float().to(self.device).float()
        m_lens = m_lens.detach().long().to(self.device)

        if self.vq_model.tcc_flag:
            out, loss_commit, perplexity, loss_tcc = self.vq_model(motions)
        else:
            out, loss_commit, perplexity =self.vq_model(motions)

        pred_motion = self.foot_refine_model(out)
        
        
        self.motions = motions
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(pred_motion, motions)
        pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)
        
        loss = loss_rec + self.opt.loss_vel * loss_explicit
        return loss, loss_rec, loss_explicit

        # return loss, loss_rec, loss_vel, loss_commit, perplexity
        # return loss, loss_rec, loss_percept, loss_commit, perplexity
        


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_fr_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "foot_refine_model": self.foot_refine_model.state_dict(),
            "opt_fr_model": self.opt_fr_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.foot_refine_model.load_state_dict(checkpoint['foot_refine_model'])
        self.opt_fr_model.load_state_dict(checkpoint['opt_fr_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']
    
    def update(self, batch_data):
        loss, loss_rec, loss_vel = self.forward(batch_data)

        self.opt_fr_model.zero_grad()
        loss.backward()
        self.opt_fr_model.step()
        self.scheduler.step()

        return loss.item(), loss_rec.item(), loss_vel.item()

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, logger, writer, plot_eval=None):
        self.foot_refine_model.to(self.device)
        self.vq_model.to(self.device)

        self.opt_fr_model = optim.AdamW(self.foot_refine_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_fr_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            logger.info("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        logger.info(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        logger.info('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

        # sys.exit()
        epoch = 0
        it = 0
        best_epoch, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger, writer = evaluation_frb(
            self.opt.out_dir, eval_val_loader, self.vq_model, self.foot_refine_model, logger, writer, ep=0, best_epoch=0, best_fid=1000,
            best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper, save=False)

        avg_loss, avg_recons, avg_vel, lr = 0., 0., 0., 0.

        for epoch in range(1, self.opt.max_epoch + 1):
            self.foot_refine_model.train()
            self.vq_model.eval()

            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, loss_rec, loss_vel = self.update(batch_data=batch)
                avg_loss += loss
                avg_recons += loss_rec
                avg_vel += loss_vel

                if it % self.opt.print_iter == 0:
                    avg_loss /= self.opt.print_iter
                    avg_recons /= self.opt.print_iter
                    avg_vel /= self.opt.print_iter
                    writer.add_scalar('./Train/Loss', avg_loss, it)
                    writer.add_scalar('./Train/Loss_rec', avg_recons, it)
                    writer.add_scalar('./Train/Loss_vel', avg_vel, it)
                    message = f"Epoch:{epoch:4d}\t Iter:{it:6d}\t lr:{self.opt_fr_model.param_groups[0]['lr']:.6f}\t loss:{avg_loss:.5f}\t rec:{avg_recons:.5f}\t vel:{avg_vel:.5f}"
                    logger.info(message)
                    avg_loss, avg_recons, avg_vel, avg_lr = 0., 0., 0., 0.

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.out_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.out_dir, 'latest.tar'), epoch, it)
            

            logger.info('Validation time:')
            self.vq_model.eval()
            self.foot_refine_model.eval()

            val_loss = []
            val_rec = []
            val_vel = []
            with torch.no_grad():
                for i, batch_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                    loss, loss_rec, loss_vel = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_rec.append(loss_rec.item())
                    val_vel.append(loss_vel.item())


            logger.info(f"Validation loss:{np.mean(val_loss):.3f}, rec:{np.mean(val_rec):.3f}, vel:{np.mean(val_vel):.3f}")

            writer.add_scalar('Val/loss', np.mean(val_loss), epoch)
            writer.add_scalar('Val/rec', np.mean(val_rec), epoch)
            writer.add_scalar('Val/vel', np.mean(val_vel), epoch)


            best_epoch, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger, writer = evaluation_frb(
                self.opt.out_dir, eval_val_loader, self.vq_model, self.foot_refine_model, logger, writer, epoch, best_epoch=best_epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1,
                best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)
            
            if epoch == self.opt.max_epoch :
                msg_final = f"Train. Epoch {best_epoch} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}, Matching. {best_matching:.4f}"
                logger.info(msg_final)

        
