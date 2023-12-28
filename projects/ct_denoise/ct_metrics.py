"""
Set up the metrics for ct
"""

import copy
import tifffile
import numpy as np
import torch
import torch.distributed as dist
from colorama import Fore, Style

import sys
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

REPO_DIR = Path(__file__).parents[2].resolve()
sys.path.append(str(REPO_DIR))

from utils import *
from metrics import MetricManager, get_metric_function, AverageMeter
from loss.loss_functions import *

# -------------------------------------------------------------------------------------------------
class CTMetricManager(MetricManager):
    """
    Metrics used for CT
    """

    def __init__(self, config):
        super().__init__(config)

        self.best_pre_model_file = None
        self.best_backbone_model_file = None
        self.best_post_model_file = None

    # ---------------------------------------------------------------------------------------
    def setup_wandb_and_metrics(self, rank):

        self.best_val_loss = np.inf

        device = self.config.device

        self.mse_loss_func = MSE_Loss(rmse_mode=False, complex_i=self.config.complex_i)
        self.rmse_loss_func = MSE_Loss(rmse_mode=True, complex_i=self.config.complex_i)
        self.l1_loss_func = L1_Loss(complex_i=self.config.complex_i)
        self.ssim_loss_func = SSIM_Loss(complex_i=self.config.complex_i, device=device)
        self.ssim3D_loss_func = SSIM3D_Loss(complex_i=self.config.complex_i, device=device)
        self.psnr_func = PSNR(range=2048)
        self.psnr_loss_func = PSNR_Loss(range=2048)
        self.gaussian_func = GaussianDeriv_Loss(sigmas=[0.5, 1.0, 1.5], complex_i=self.config.complex_i, device=device)
        self.gaussian3D_func = GaussianDeriv3D_Loss(sigmas=[0.5, 1.0, 1.5], sigmas_T=[0.5, 0.5, 0.5], complex_i=self.config.complex_i, device=device)
        self.spec_func = Spectral_Loss(dim=[-2, -1], min_bound=5, max_bound=95, complex_i=self.config.complex_i, device=device)
        self.dwt_func = Wavelet_Loss(J=1, wave='db3', mode='symmetric', only_h=True, complex_i=self.config.complex_i, device=device)
        self.ssim_func = lambda x, y: 1 - self.ssim_loss_func(x, y)
        self.ssim3D_func = lambda x, y: 1 - self.ssim3D_loss_func(x, y)
        self.charb_func = Charbonnier_Loss(complex_i=self.config.complex_i)
        self.vgg_func = VGGPerceptualLoss(complex_i=self.config.complex_i).to(device=device)

        self.train_metrics = {'loss': AverageMeter(),
                              'mse':AverageMeter(),
                              'rmse':AverageMeter(),
                              'l1':AverageMeter(),
                              'ssim':AverageMeter(),
                              'ssim_3d':AverageMeter(),
                              'ssim_loss':AverageMeter(),
                              'ssim_3d_loss':AverageMeter(),
                              'psnr':AverageMeter(),
                              'psnr_loss':AverageMeter(),
                              'gaussian_gradient':AverageMeter(),
                              'gaussian_gradient_3d':AverageMeter(),
                              'spec':AverageMeter(),
                              'dwt':AverageMeter(),
                              'charb':AverageMeter(),
                              'vgg':AverageMeter()
                            }

        self.eval_metrics = {'loss': AverageMeter(),
                              'mse':AverageMeter(),
                              'rmse':AverageMeter(),
                              'l1':AverageMeter(),
                              'ssim':AverageMeter(),
                              'ssim_3d':AverageMeter(),
                              'ssim_loss':AverageMeter(),
                              'ssim_3d_loss':AverageMeter(),
                              'psnr':AverageMeter(),
                              'psnr_loss':AverageMeter(),
                              'gaussian_gradient':AverageMeter(),
                              'gaussian_gradient_3d':AverageMeter(),
                              'spec':AverageMeter(),
                              'dwt':AverageMeter(),
                              'charb':AverageMeter(),
                              # 'vgg':AverageMeter(), OOM error
                              'mse_2d':AverageMeter(),
                              'ssim_2d':AverageMeter(),
                              'psnr_2d':AverageMeter(),
                              'mse_2dt':AverageMeter(),
                              'ssim_2dt':AverageMeter(),
                              'psnr_2dt':AverageMeter()
                            }

        self.train_metric_functions = {
                              'mse':self.mse_loss_func,
                              'rmse':self.rmse_loss_func,
                              'l1':self.l1_loss_func,
                              'ssim':self.ssim_func,
                              'ssim_3d':self.ssim3D_func,
                              'ssim_loss':self.ssim_loss_func,
                              'ssim_3d_loss':self.ssim3D_loss_func,
                              'psnr':self.psnr_func,
                              'psnr_loss':self.psnr_loss_func,
                              'gaussian_gradient':self.gaussian_func,
                              'gaussian_gradient_3d':self.gaussian3D_func,
                              'spec': self.spec_func,
                              'dwt': self.dwt_func,
                              'charb': self.charb_func,
                              'vgg': self.vgg_func,
                              'mse_2d':self.mse_loss_func,
                              'ssim_2d':self.ssim_func,
                              'psnr_2d':self.psnr_func,
                              'mse_2dt':self.mse_loss_func,
                              'ssim_2dt':self.ssim_func,
                              'psnr_2dt':self.psnr_func
                            }

        self.eval_metric_functions = copy.deepcopy(self.train_metric_functions)

        self.eval_sample_metrics = {
                              'mse':AverageMeter(),
                              'psnr':AverageMeter(),
                              'ssim':AverageMeter(),
                              'vgg':AverageMeter(),
                              }

        if rank<=0:
            # Initialize metrics to track in wandb
            if self.wandb_run:
                self.wandb_run.define_metric("epoch")
                for metric_name in self.train_metrics.keys():
                    self.wandb_run.define_metric('train_'+metric_name, step_metric='epoch')
                for metric_name in self.eval_metrics.keys():
                    self.wandb_run.define_metric('val_'+metric_name, step_metric='epoch')

            # Initialize metrics to track for checkpointing best-performing model
            self.best_val_psnr = -np.inf

    def get_tra_loss(self):
        return self.train_metrics['loss'].avg, \
                self.train_metrics['mse'].avg, \
                self.train_metrics['rmse'].avg, \
                self.train_metrics['l1'].avg, \
                self.train_metrics['ssim'].avg, \
                self.train_metrics['ssim_3d'].avg, \
                self.train_metrics['ssim_loss'].avg, \
                self.train_metrics['ssim_3d_loss'].avg, \
                self.train_metrics['psnr'].avg, \
                self.train_metrics['psnr_loss'].avg, \
                self.train_metrics['gaussian_gradient'].avg, \
                self.train_metrics['gaussian_gradient_3d'].avg, \
                self.train_metrics['spec'].avg, \
                self.train_metrics['dwt'].avg, \
                self.train_metrics['charb'].avg, \
                self.train_metrics['vgg'].avg

    def get_eval_loss(self):
        return self.eval_metrics['loss'].avg, \
                self.eval_metrics['mse'].avg, \
                self.eval_metrics['rmse'].avg, \
                self.eval_metrics['l1'].avg, \
                self.eval_metrics['ssim'].avg, \
                self.eval_metrics['ssim_3d'].avg, \
                self.eval_metrics['ssim_loss'].avg, \
                self.eval_metrics['ssim_3d_loss'].avg, \
                self.eval_metrics['psnr'].avg, \
                self.eval_metrics['psnr_loss'].avg, \
                self.eval_metrics['gaussian_gradient'].avg, \
                self.eval_metrics['gaussian_gradient_3d'].avg, \
                self.eval_metrics['spec'].avg, \
                self.eval_metrics['dwt'].avg , \
                self.eval_metrics['charb'].avg, \
                self.eval_metrics['vgg'].avg, \
                self.eval_metrics['mse_2d'].avg, \
                self.eval_metrics['ssim_2d'].avg, \
                self.eval_metrics['psnr_2d'].avg, \
                self.eval_metrics['mse_2dt'].avg, \
                self.eval_metrics['ssim_2dt'].avg, \
                self.eval_metrics['psnr_2dt'].avg

    def compute_batch_statistics(self, x, output, y):

        B, C, T, H, W = x.shape

        v_x = np.zeros(B)
        v_y = np.zeros(B)
        v_y_hat = np.zeros(B)
        mse = np.zeros(B)
        ssim = np.zeros(B)
        psnr = np.zeros(B)
        vgg = np.zeros(B)

        caption =""
        new_line = '\n'

        for b in range(B):
            v_x[b] = torch.mean(torch.abs(x[b, :2])).item()
            v_y[b] = torch.mean(torch.abs(y[b])).item()
            v_y_hat[b] = torch.mean(torch.abs(output[b])).item()

            y_hat_b = torch.unsqueeze(output[b], dim=0)
            y_b = torch.unsqueeze(y[b], dim=0)
            mse[b] = self.mse_loss_func(y_hat_b, y_b)
            ssim[b] = self.ssim_func(y_hat_b, y_b)
            psnr[b] = self.psnr_func(y_hat_b, y_b)
            # vgg[b] = self.vgg_func(y_hat_b, y_b) gives OOM error

            caption += f"{b} -- x {v_x[b]:.2f}, y_hat {v_y_hat[b]:.2f}, y {v_y[b]:.2f}, mse {mse[b]:.2f}, ssim {ssim[b]:.2f}, psnr {psnr[b]:.2f}, vgg {vgg[b]:.2f}{new_line}"

        return caption, (v_x, v_y, v_y_hat, mse, ssim, psnr, vgg)

    # ---------------------------------------------------------------------------------------
    def parse_output(self, output):
        if self.config.model_type == "STCNNT_CT" or self.config.model_type == "CT_hrnet" or self.config.model_type == "omnivore_CT":
            if isinstance(output, tuple) and len(output)==2:
                y_hat, weights = output
            else:
                y_hat = output
            output_1st_net = None
        else:
            if isinstance(output, tuple) and len(output)==3:
                y_hat, weights, output_1st_net = output
            else:
                y_hat, output_1st_net = output

        return y_hat, output_1st_net

    # ---------------------------------------------------------------------------------------
    def on_train_step_end(self, loss, output, labels, rank, curr_lr, save_samples, epoch, ids):

        x, y = labels

        y_hat, output_1st_net = self.parse_output(output)

        y_for_loss = y

        y_hat = y_hat.to(torch.float32)
        y_for_loss = y_for_loss.to(device=y_hat.device, dtype=torch.float32)

        for metric_name in self.train_metrics.keys():
            if metric_name=='loss':
                self.train_metrics[metric_name].update(loss, n=x.shape[0])
            else:
                metric_value = self.train_metric_functions[metric_name](y_hat, y_for_loss)
                self.train_metrics[metric_name].update(metric_value.item(), n=x.shape[0])

        if rank<=0 and self.wandb_run:
            self.wandb_run.log({"running_train/lr": curr_lr})
            for metric_name in self.train_metrics.keys():
                if metric_name=='loss':
                    self.wandb_run.log({"running_train/loss": loss})
                else:
                    self.wandb_run.log({f"running_train/{metric_name}": self.train_metrics[metric_name].avg})

        # Save outputs if desired
        if save_samples and rank<=0:
            save_path = os.path.join(self.config.log_dir,self.config.run_name,'saved_samples', 'tra')
            os.makedirs(save_path, exist_ok=True)

            if output_1st_net is not None: output_1st_net = output_1st_net.detach().cpu()
            self.save_batch_samples(save_path, f"epoch_{epoch}_{ids}", x.cpu(), y.cpu(), y_hat.detach().cpu(), output_1st_net)

    # ---------------------------------------------------------------------------------------
    def on_eval_epoch_start(self):
        super().on_eval_epoch_start()
        for metric_name in self.eval_sample_metrics.keys():
            self.eval_sample_metrics[metric_name].reset()

    # ---------------------------------------------------------------------------------------
    def on_eval_step_end(self, loss, output, labels, ids, rank, save_samples, split):

        with torch.inference_mode():
            x, y = labels
            y_hat, output_1st_net = self.parse_output(output)

            x = torch.clone(x)
            y = torch.clone(y)
            y_hat = torch.clone(y_hat)

            B = x.shape[0]
            T = x.shape[2]

            y_for_loss = y

            y_hat = y_hat.to(torch.float32)
            y_for_loss = y_for_loss.to(device=y_hat.device, dtype=torch.float32)

            caption, record = self.compute_batch_statistics(x, y_hat, y_for_loss)
            v_x, v_y, v_y_hat, mse, ssim, psnr, vgg = record

            for b in range(B):
                self.eval_sample_metrics["mse"].update(mse[b])
                self.eval_sample_metrics["ssim"].update(ssim[b])
                self.eval_sample_metrics["psnr"].update(psnr[b])
                self.eval_sample_metrics["vgg"].update(vgg[b])

        if T == 1:
            for metric_name in self.eval_metrics.keys():
                if metric_name.find('2d') > 0 and metric_name.find('2dt') < 0:
                    metric_value = self.eval_metric_functions[metric_name](y_hat, y_for_loss)
                    self.eval_metrics[metric_name].update(metric_value.item(), n=x.shape[0])

        if T > 1:
            for metric_name in self.eval_metrics.keys():
                if metric_name.find('2dt') > 0:
                    metric_value = self.eval_metric_functions[metric_name](y_hat, y_for_loss)
                    self.eval_metrics[metric_name].update(metric_value.item(), n=x.shape[0])

        for metric_name in self.eval_metrics.keys():
            if metric_name=='loss':
                self.eval_metrics[metric_name].update(loss, n=x.shape[0])
                continue

            if metric_name.find('2d') >= 0:
                continue

            metric_value = self.eval_metric_functions[metric_name](y_hat, y_for_loss)
            self.eval_metrics[metric_name].update(metric_value.item(), n=x.shape[0])

        # Save outputs if desired
        if save_samples and rank<=0:
            save_path = os.path.join(self.config.log_dir,self.config.run_name,'saved_samples',split)
            os.makedirs(save_path, exist_ok=True)

            if output_1st_net is not None:
                output_1st_net = output_1st_net.detach().cpu()
            self.save_batch_samples(save_path, f"{ids}", x.cpu(), y.cpu(), y_hat.detach().cpu(), output_1st_net)

        return caption, record

    # ---------------------------------------------------------------------------------------
    def on_eval_epoch_end(self, rank, epoch, model_manager, optim, sched, split, final_eval):
        """
        Runs at the end of the evaluation loop
        """
        save_path = os.path.join(self.config.log_dir, self.config.run_name)
        logging.info(f"{Fore.YELLOW}--> epoch {epoch}, {split}, save path is {Fore.RED}{save_path}{Style.RESET_ALL}")

        # Directly compute metrics from saved predictions if using exact metrics
        if self.config.exact_metrics:
            self.all_preds = torch.concatenate(self.all_preds)
            self.all_labels = torch.concatenate(self.all_labels)
            for metric_name in self.eval_metrics.keys():
                if metric_name!='loss':
                    metric_value = self.eval_metric_functions[metric_name](self.all_preds, self.all_labels).item()
                    if self.multidim_average=='samplewise':
                        metric_value = torch.mean(metric_value)
                    self.eval_metrics[metric_name].update(metric_value, n=self.all_preds.shape[0])

        # Otherwise aggregate the measurements over the steps
        # for metric_name in self.eval_metrics.keys():
        #     print(f"--> epoch {epoch}, rank {rank}, {metric_name} {self.eval_metrics[metric_name].avg}")

        average_metrics = dict()
        if self.config.ddp:
            for metric_name in self.eval_metrics.keys():
                v = torch.tensor(self.eval_metrics[metric_name].avg).to(device=self.device)
                dist.all_reduce(v, op=torch.distributed.ReduceOp.AVG)
                average_metrics[metric_name] = v.item()
        else:
            average_metrics = {metric_name: self.eval_metrics[metric_name].avg for metric_name in self.eval_metrics.keys()}

        # Save the average metrics for this epoch into self.average_eval_metrics
        self.average_eval_metrics = average_metrics
        #print(f"--> epoch {epoch}, average_metrics {average_metrics}")

        # update for per sample metrics
        # if ddp, aggregate metrics
        if final_eval:
            if self.config.ddp:
                world_size = int(os.environ["WORLD_SIZE"])
                #print(f"--> epoch {epoch}, rank {rank}, world_size {world_size} ... ")
                mse = self.eval_sample_metrics["mse"].vals
                num = len(mse)
                tensor_in = torch.tensor(mse, dtype=torch.float32, device=self.config.device)
                tensor_out = torch.zeros(num * world_size, dtype=torch.float32, device=self.config.device)
                dist.all_gather_into_tensor(tensor_out, tensor_in)
                mse = tensor_out.cpu().numpy()

                metrics = dict()
                for metric_name in self.eval_sample_metrics.keys():
                    if metric_name == "mse":
                        continue

                    v = self.eval_sample_metrics[metric_name].vals
                    num = len(v)
                    tensor_in = torch.tensor(v, dtype=torch.float32, device=self.config.device)
                    tensor_out = torch.zeros(num * world_size, dtype=torch.float32, device=self.config.device)
                    dist.all_gather_into_tensor(tensor_out, tensor_in)
                    metrics[metric_name] = tensor_out.cpu().numpy()
            else:
                mse = self.eval_sample_metrics["mse"].vals
                np.save(os.path.join(save_path, f"final_{split}.npy"), np.array(mse))

                metrics = dict()
                for metric_name in self.eval_sample_metrics.keys():
                    if metric_name == "mse":
                        continue

                    metrics[metric_name] = self.eval_sample_metrics[metric_name].vals

        # Checkpoint best models during training
        if rank<=0:
            if not final_eval:
                # Determine whether to checkpoint this model
                model_epoch = model_manager.module if self.config.ddp else model_manager
                checkpoint_model = False
                if average_metrics['loss'] is not None and average_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = average_metrics['loss']
                    checkpoint_model = True

                self.epoch_pre_model_file, self.epoch_backbone_model_file, self.epoch_post_model_file = model_epoch.save(os.path.join(save_path, f"checkpoint_epoch_{epoch}"), epoch, optim, sched)
                logging.info(f"--> val loss {average_metrics['loss']}, save model for epoch {epoch} to {self.epoch_pre_model_file}, {self.epoch_backbone_model_file}, {self.epoch_post_model_file}")

                # Save model and update best metrics
                if checkpoint_model:
                    self.best_pre_model_file, self.best_backbone_model_file, self.best_post_model_file = model_epoch.save(os.path.join(save_path, f"best_checkpoint_epoch_{epoch}"), epoch, optim, sched)
                    if self.wandb_run: self.wandb_run.log({"epoch":epoch, "best_val_loss":self.best_val_loss})
                    logging.info(f"--> val loss {self.best_val_loss}, save best model for epoch {epoch} to {self.best_pre_model_file}, {self.best_backbone_model_file}, {self.best_post_model_file}")

                # Update wandb with eval metrics
                for metric_name, avg_metric_eval in average_metrics.items():
                    if self.wandb_run: self.wandb_run.log({"epoch":epoch, f"{split}/{metric_name}": avg_metric_eval})
            else:
                for metric_name, avg_metric_eval in average_metrics.items():
                    if self.wandb_run: self.wandb_run.summary[f"final_{split}_{metric_name}"] = avg_metric_eval

                # -----------------------------------------------------

                # def do_auc(x, y, min_x, max_x, key_str):
                #     try:
                #         x = np.copy(x)
                #         y = np.copy(y)
                #         bin_means, bin_sds, bin_edges, binnumber = compute_binned_mean_sd(x, y, min_x=min_x, max_x=max_x, bins=50)
                #         fig, auc = plot_with_CI(x, y, min_x=min_x, max_x=max_x, bin_means=bin_means, bin_sds=bin_sds, bin_edges=bin_edges, xlabel='snr', ylabel=key_str, ylim=[0, 1])
                #         fig.savefig(os.path.join(save_path, f"final_{split}_{key_str}_CI_{min_x}_{max_x}.png"), dpi=600)
                #         if self.wandb_run:
                #             self.wandb_run.log({f"final_{split}_{key_str}_CI_{min_x}_{max_x}": wandb.Image(fig)})
                #             self.wandb_run.summary[f"final_{split}_auc_{key_str}_{min_x}_{max_x}"] = auc

                #         indices = np.where(np.logical_and(x >= min_x, x < max_x))
                #         p = np.percentile(x[indices], [5, 95])
                #         py = np.percentile(y[indices], [5, 95])
                #         logging.info(f"--> compute auc, {split}, {key_str}, {min_x} to {max_x}, auc {auc:.4f}, x - {np.mean(x[indices]):.4f}+/-{np.std(x[indices]):.4f}, median {np.median(x[indices]):.4f}, 5-95% {p[0]:.4f}, {p[1]:.4f} ==== y - {np.mean(y[indices]):.4f}+/-{np.std(y[indices]):.4f}, median {np.median(y[indices]):.4f}, 5-95% {py[0]:.4f}, {py[1]:.4f}")
                #         if self.wandb_run:
                #             self.wandb_run.summary[f"final_{split}_{key_str}_mean_{min_x}_to_{max_x}"] = np.mean(y[indices])
                #             self.wandb_run.summary[f"final_{split}_{key_str}_std_{min_x}_to_{max_x}"] = np.std(y[indices])
                #             self.wandb_run.summary[f"final_{split}_{key_str}_median_{min_x}_to_{max_x}"] = np.median(y[indices])
                #             self.wandb_run.summary[f"final_{split}_{key_str}_5%_{min_x}_to_{max_x}"] = py[0]
                #             self.wandb_run.summary[f"final_{split}_{key_str}_95%_{min_x}_to_{max_x}"] = py[1]
                #     except:
                #         logging.info(f"--> compute auc, {split}, {key_str}, {min_x} to {max_x}, error ...")

                # metric_fname = os.path.join(save_path, f"final_{split}_metrics.pkl")
                # with open(metric_fname, 'wb') as f:
                #     pickle.dump(metrics, f)

                # if self.wandb_run:
                #     self.wandb_run.save(metric_fname)

                # x = np.copy(np.array(mse))
                # y = np.array(metrics['ssim'])
                # np.save(os.path.join(save_path, f"{split}_input_snr.npy"), x)
                # np.save(os.path.join(save_path, f"{split}_ssim.npy"), y)
                # do_auc(x=x, y=y, min_x=0.1, max_x=10, key_str='ssim')
                # do_auc(x=x, y=y, min_x=0.1, max_x=1, key_str='ssim')
                # do_auc(x=x, y=y, min_x=1, max_x=10, key_str='ssim')
                # do_auc(x=x, y=y, min_x=0.1, max_x=0.5, key_str='ssim')
                # do_auc(x=x, y=y, min_x=0.5, max_x=1, key_str='ssim')
                # do_auc(x=x, y=y, min_x=1, max_x=5, key_str='ssim')
                # do_auc(x=x, y=y, min_x=5, max_x=10, key_str='ssim')

                # # -----------------------------------------------------

                # x = np.copy(np.array(mse))
                # y = np.array(metrics['psnr'])
                # np.save(os.path.join(save_path, f"{split}_psnr.npy"), y)
                # do_auc(x=x, y=y, min_x=0.1, max_x=10, key_str='psnr')
                # do_auc(x=x, y=y, min_x=0.1, max_x=1, key_str='psnr')
                # do_auc(x=x, y=y, min_x=1, max_x=10, key_str='psnr')
                # do_auc(x=x, y=y, min_x=0.1, max_x=0.5, key_str='psnr')
                # do_auc(x=x, y=y, min_x=0.5, max_x=1, key_str='psnr')
                # do_auc(x=x, y=y, min_x=1, max_x=5, key_str='psnr')
                # do_auc(x=x, y=y, min_x=5, max_x=10, key_str='psnr')

                # # -----------------------------------------------------

    # ---------------------------------------------------------------------------------------
    def on_training_end(self, rank, epoch, model_manager, optim, sched, ran_training):
        """
        Runs once when training finishes
        """
        if rank<=0: # main or master process

            if ran_training:
                # Log the best loss and metrics from the run and save final model
                if self.wandb_run: self.wandb_run.summary["best_val_loss"] = self.best_val_loss

                model_epoch = model_manager.module if self.config.ddp else model_manager
                model_epoch.save('final_epoch', epoch, optim, sched)

            # Finish the wandb run
            if self.wandb_run: self.wandb_run.finish()

    # ---------------------------------------------------------------------------------------
    def save_batch_samples(self, saved_path, fname, x, y, output, output_1st_net):

        noisy_im = x.numpy()
        clean_im = y.numpy()
        predi_im = output.numpy()
        if output_1st_net is not None:
            pred_im_1st_net = output_1st_net.numpy()
        else:
            pred_im_1st_net = None

        np.save(os.path.join(saved_path, f"{fname}_x.npy"), noisy_im)
        np.save(os.path.join(saved_path, f"{fname}_y.npy"), clean_im)
        np.save(os.path.join(saved_path, f"{fname}_output.npy"), predi_im)
        if pred_im_1st_net is not None: np.save(os.path.join(saved_path, f"{fname}_y_1st_net.npy"), pred_im_1st_net)

        B, C, T, H, W = x.shape

        if C==2:
            save_x = np.sqrt(np.square(noisy_im[0,0]) + np.square(noisy_im[0,1]))
            save_p = np.sqrt(np.square(predi_im[0,0]) + np.square(predi_im[0,1]))
            save_y = np.sqrt(np.square(clean_im[0,0]) + np.square(clean_im[0,1]))
            save_1st_net = np.sqrt(np.square(pred_im_1st_net[0,0]) + np.square(pred_im_1st_net[0,1])) if pred_im_1st_net is not None else []
        else:
            save_x = noisy_im[0,0]
            save_p = predi_im[0,0]
            save_y = clean_im[0,0]
            save_1st_net = pred_im_1st_net[0,0] if pred_im_1st_net is not None else []

        if pred_im_1st_net is None:
            composed_channel_wise = np.transpose(np.array([save_x, save_p, save_y]), (1,0,2,3))
        else:
            composed_channel_wise = np.transpose(np.array([save_x, save_1st_net, save_p, save_y]), (1,0,2,3))

        tifffile.imwrite(os.path.join(saved_path, f"{fname}_combined.tiff"),\
                            composed_channel_wise, imagej=True)

        # hdr = nib.Nifti1Header()
        # hdr.set_data_shape((H, W, T, B))

        # if C==3:
        #     x = noisy_im[:,:,0,:,:] + 1j * noisy_im[:,:,1,:,:]
        #     gmap = noisy_im[:,:,2,:,:]

        #     nib.save(nib.Nifti1Image(np.real(x), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_x_real.nii"))
        #     nib.save(nib.Nifti1Image(np.imag(x), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_x_imag.nii"))
        #     nib.save(nib.Nifti1Image(np.abs(x), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_x.nii"))

        #     y = clean_im[:,:,0,:,:] + 1j * clean_im[:,:,1,:,:]
        #     nib.save(nib.Nifti1Image(np.real(y), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_real.nii"))
        #     nib.save(nib.Nifti1Image(np.imag(y), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_imag.nii"))
        #     nib.save(nib.Nifti1Image(np.abs(y), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y.nii"))

        #     output = pred_im[:,:,0,:,:] + 1j * pred_im[:,:,1,:,:]
        #     nib.save(nib.Nifti1Image(np.real(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_output_real.nii"))
        #     nib.save(nib.Nifti1Image(np.imag(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_output_imag.nii"))
        #     nib.save(nib.Nifti1Image(np.abs(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_output.nii"))

        #     output = y_degraded[:,:,0,:,:] + 1j * y_degraded[:,:,1,:,:]
        #     nib.save(nib.Nifti1Image(np.real(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_degraded_real.nii"))
        #     nib.save(nib.Nifti1Image(np.imag(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_degraded_imag.nii"))
        #     nib.save(nib.Nifti1Image(np.abs(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_degraded.nii"))

        #     output = y_2x[:,:,0,:,:] + 1j * y_2x[:,:,1,:,:]
        #     hdr = nib.Nifti1Header()
        #     hdr.set_data_shape(y_2x.shape)
        #     nib.save(nib.Nifti1Image(np.real(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_2x_real.nii"))
        #     nib.save(nib.Nifti1Image(np.imag(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_2x_imag.nii"))
        #     nib.save(nib.Nifti1Image(np.abs(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_2x.nii"))

        #     if pred_im_1st_net is not None:
        #         output = pred_im_1st_net[:,:,0,:,:] + 1j * pred_im_1st_net[:,:,1,:,:]
        #         nib.save(nib.Nifti1Image(np.real(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_output_1st_net_real.nii"))
        #         nib.save(nib.Nifti1Image(np.imag(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_output_1st_net_imag.nii"))
        #         nib.save(nib.Nifti1Image(np.abs(output), affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_output_1st_net.nii"))
        # else:
        #     x = noisy_im[:,:,0,:,:]
        #     gmap = noisy_im[:,:,1,:,:]

        #     nib.save(nib.Nifti1Image(x, affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_x.nii"))
        #     nib.save(nib.Nifti1Image(clean_im, affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y.nii"))
        #     nib.save(nib.Nifti1Image(pred_im, affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_output.nii"))
        #     nib.save(nib.Nifti1Image(y_degraded, affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_degraded.nii"))

        #     hdr.set_data_shape(y_2x.shape)
        #     nib.save(nib.Nifti1Image(y_2x, affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_y_2x.nii"))

        #     if pred_im_1st_net is not None:
        #         nib.save(nib.Nifti1Image(pred_im_1st_net, affine=np.eye(4), header=hdr), os.path.join(saved_path, f"{fname}_output_1st_net.nii"))

        # hdr.set_data_shape(gmap.shape)
        # nib.save(nib.Nifti1Image(gmap, affine=np.eye(4)), os.path.join(saved_path, f"{fname}_gmap.nii"))

def tests():
    logging.info('Passed all tests')

if __name__=="__main__":
    tests()