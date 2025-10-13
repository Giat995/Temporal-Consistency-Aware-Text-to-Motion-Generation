import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *

from einops import rearrange, repeat
from tqdm import tqdm

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()



    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        code_idx, _ = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []

        _loss, _pred_ids, _acc = self.t2m_transformer(code_idx[..., 0], conds, m_lens)

        return _loss, _acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, logger, writer, plot_eval):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            logger.info("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        logger.info(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        logger.info('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        
        epoch = 0
        best_epoch, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger, writer = evaluation_mask_transformer(
            self.opt.out_dir, eval_val_loader, self.t2m_transformer, self.vq_model, logger, writer, epoch, best_epoch=0, 
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_acc = 0.

        avg_loss, avg_acc, avg_lr = 0., 0., 0.

        for epoch in range(1, self.opt.max_epoch + 1):
            self.t2m_transformer.train()
            self.vq_model.eval()

            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                avg_loss += loss
                avg_acc += acc

                if it % self.opt.print_iter == 0:
                    avg_loss /= self.opt.print_iter
                    avg_acc /= self.opt.print_iter
                    writer.add_scalar('./Train/Loss', avg_loss, it)
                    writer.add_scalar('./Train/Acc', avg_acc, it)
                    message = f"Epoch:{epoch:4d}\t Iter:{it:6d}\t lr:{self.opt_t2m_transformer.param_groups[0]['lr']:.6f}\t loss:{avg_loss:.5f}\t acc:{avg_acc:.5f}"
                    logger.info(message)
                    avg_loss, avg_acc, avg_lr = 0., 0., 0.

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.out_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.out_dir, 'latest.tar'), epoch, it)
            

            logger.info('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            logger.info(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            writer.add_scalar('Val/loss', np.mean(val_loss), epoch)
            writer.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > best_acc:
                logger.info(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc):.02f}!!!")
                self.save(pjoin(self.opt.out_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_epoch, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger, writer = evaluation_mask_transformer(
                self.opt.out_dir, eval_val_loader, self.t2m_transformer, self.vq_model, logger, writer, epoch, best_epoch = best_epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=True, save_anim=False
            )
            
            if epoch == self.opt.max_epoch :
                msg_final = f"Train. Epoch {best_epoch} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}, Matching. {best_matching:.4f}"
                logger.info(msg_final)


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()



    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        code_idx, all_codes = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens)

        return ce_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        res_trans_state_dict = self.res_transformer.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, logger, writer, plot_eval):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_res_transformer = optim.AdamW(self.res_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            logger.info("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        logger.info(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        logger.info('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

        epoch = 0
        best_epoch, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger, writer = evaluation_res_transformer(
            self.opt.out_dir, eval_val_loader, self.res_transformer, self.vq_model, logger, writer, epoch, best_epoch=0, 
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_loss = 100
        best_acc = 0

        avg_loss, avg_acc, avg_lr = 0., 0., 0.

        for epoch in range(1, self.opt.max_epoch + 1):
            self.res_transformer.train()
            self.vq_model.eval()

            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                avg_loss += loss
                avg_acc += acc

                if it % self.opt.print_iter == 0:
                    avg_loss /= self.opt.print_iter
                    avg_acc /= self.opt.print_iter
                    writer.add_scalar('./Train/Loss', avg_loss, it)
                    writer.add_scalar('./Train/Acc', avg_acc, it)
                    message = f"Epoch:{epoch:4d}\t Iter:{it:6d}\t lr:{self.opt_res_transformer.param_groups[0]['lr']:.6f}\t loss:{avg_loss:.5f}\t acc:{avg_acc:.5f}"
                    logger.info(message)
                    avg_loss, avg_acc, avg_lr = 0., 0., 0.

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.out_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.out_dir, 'latest.tar'), epoch, it)

            logger.info('Validation time:')
            self.vq_model.eval()
            self.res_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            logger.info(f"Validation loss:{np.mean(val_loss):.3f}, Accuracy:{np.mean(val_acc):.3f}")

            writer.add_scalar('Val/loss', np.mean(val_loss), epoch)
            writer.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_loss) < best_loss:
                logger.info(f"Improved loss from {best_loss:.02f} to {np.mean(val_loss)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_loss = np.mean(val_loss)

            if np.mean(val_acc) > best_acc:
                logger.info(f"Improved acc from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                # self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_epoch, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger, writer = evaluation_res_transformer(
                self.opt.out_dir, eval_val_loader, self.res_transformer, self.vq_model, logger, writer, epoch, best_epoch=best_epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            )

            if epoch == self.opt.max_epoch :
                msg_final = f"Train. Epoch {best_epoch} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}, Matching. {best_matching:.4f}"
                logger.info(msg_final)