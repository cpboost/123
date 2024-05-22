
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
import sys
sys.path.append('/data01/dyf/CaPaint')
from models.mau_model import MAU_Model
from omegaconf import OmegaConf
from tqdm import tqdm
from API import *
from utils import *


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()



    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        
       # config.py
        class Configs:
            def __init__(self):
                # 定义所有必要的参数
                self.in_shape = tuple(args.in_shape)  # (T, C, H, W)
                self.patch_size = 1
                self.sr_size = 4
                self.filter_size = 5
                self.stride = 1
                self.tau = 5
                self.cell_mode = 'normal'
                self.model_mode = 'recall'
                self.layer_norm = True
                self.total_length = 10
                self.pre_seq_length = 10

        configs = Configs()
        num_layers = 1
        num_hidden = [32]

        self.model = MAU_Model(num_layers, num_hidden, configs).to(self.device)


       


    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(config['rate'],config['dataname'],config['batch_size'],config['val_batch_size'],config['data_root'],config['num_workers'])
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        #Adam
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                # print("batch_size:", batch_y.size())
                self.optimizer.zero_grad()
                B, T, C, H, W = batch_x.shape
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                mask_true = torch.ones(B, T, C, H, W).to(self.device)  # 全为1的mask
                pred_y, loss = self.model(batch_x, mask_true)
                batch_y = batch_y[:, :-1]
                #print("pred_size:", pred_y.size())
                #print("batch_size:", batch_y.size())
                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            # if i * batch_x.shape[0] > 1000:
            #     break
            B, T, C, H, W = batch_x.shape
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            mask_true = torch.ones(B, T, C, H, W).to(self.device)  # 全为1的mask
            pred_y, loss = self.model(batch_x, mask_true)
            batch_y = batch_y[:, :-1]
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        # mse, mae= metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        # print_log('vali mse:{:.4f}, mae:{:.4f}'.format(mse, mae))
        self.model.train()


        return total_loss

    def test(self, args):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            B, T, C, H, W = batch_x.shape
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            mask_true = torch.ones(B, T, C, H, W).to(self.device)  # 全为1的mask
            pred_y, lll = self.model(batch_x, mask_true)
            batch_y = batch_y[:, :-1]
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # mse, mae = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        # print_log('mse:{:.4f}, mae:{:.4f}'.format(mse, mae))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        #return mse