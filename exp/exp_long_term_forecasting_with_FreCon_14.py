from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, shape_metric
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

from statsmodels.tsa.api import acf
from models.AutoConNet import series_decomp
from layers.losses import AutoCon, AutoConCI
from layers.Dominant_Estimation import GetDominant
from layers.Frelosses_14 import DominantLoss

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_with_FreLoss(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_with_FreLoss, self).__init__(args)

        self.AutoCon = args.AutoCon
        self.AutoCon_lambda = args.AutoCon_lambda
        st = time.time()
        # self.AutoCon_loss = self.init_AutoCon(args)
        self.FreLoss = self.init_FreLoss(args)
        ed = time.time()
        print(f'Autocorrelation calculation time: {ed - st:.4f}')

    def init_FreLoss(self, args):
        target_data, _ = self._get_data(flag='train')
        target_data = target_data.data_x.copy()
        # smoother = series_decomp(args.seq_len + 1)
        dominant = GetDominant(args.global_k, first_freq=0)
        x = torch.from_numpy(target_data).unsqueeze(0)
        # _, target_data = dominant(x)
        target_data, _ = dominant(x)
        target_data = target_data.squeeze(0).numpy()
        acf_values = []
        for i_ch in range(target_data.shape[-1]):
            acf_values.append(acf(target_data[..., i_ch], nlags=len(target_data)))
        #
        # if self.args.model == 'AutoConCI':
        #     acf_values = np.stack(acf_values, axis=0)
        #     loss = AutoConCI(args.batch_size, args.seq_len, np.abs(acf_values), temperature=1.0, base_temperature=1.0)
        #     print(f'Auto-correlation values(abs):{acf_values[0, :2]} ~ {acf_values[0, -2:]}')
        # else:
        #     acf_values = np.stack(acf_values, axis=0).mean(axis=0)
        #     loss = AutoCon(args.batch_size, args.seq_len, np.abs(acf_values), temperature=1.0, base_temperature=1.0)
        #     print(f'Auto-correlation values(abs):{acf_values[:2]} ~ {acf_values[-2:]}')
        # acf_values = np.stack(acf_values, axis=0)
        acf_values = np.stack(acf_values, axis=0).mean(axis=0)
        loss = DominantLoss(args, np.abs(acf_values), temperature=1.0, base_temperature=1.0)
        return loss

    def init_AutoCon(self, args):
        target_data, _ = self._get_data(flag='train')
        target_data = target_data.data_x.copy()
        smoother = series_decomp(args.seq_len + 1)
        x = torch.from_numpy(target_data).unsqueeze(0)
        _, target_data = smoother(x)
        target_data = target_data.squeeze(0).numpy()
        acf_values = []
        for i_ch in range(target_data.shape[-1]):
            acf_values.append(acf(target_data[..., i_ch], nlags=len(target_data)))

        if self.args.model == 'AutoConCI':
            acf_values = np.stack(acf_values, axis=0)
            loss = AutoConCI(args.batch_size, args.seq_len, np.abs(acf_values), temperature=1.0, base_temperature=1.0)
            print(f'Auto-correlation values(abs):{acf_values[0, :2]} ~ {acf_values[0, -2:]}')
        else:
            acf_values = np.stack(acf_values, axis=0).mean(axis=0)
            loss = AutoCon(args.batch_size, args.seq_len, np.abs(acf_values), temperature=1.0, base_temperature=1.0)
            print(f'Auto-correlation values(abs):{acf_values[:2]} ~ {acf_values[-2:]}')
        return loss

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print(f'model parameters:{self.count_parameters(model)}')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if self.args.AutoCon:
                    # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    if self.args.model == 'DeepTime':
                        outputs, repr = self.model(batch_x, batch_x_mark, batch_y_mark)
                    else:
                        outputs, repr, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'DeepTime':
                        outputs, repr = self.model(batch_x, batch_x_mark, batch_y_mark)
                    else:
                        outputs, repr, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_log = dict()
            train_log['loss'] = []
            train_log['MSE_loss'] = []
            train_log['FreLoss'] = []
            # train_loss = []

            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                B, T, C = batch_x.shape

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if not self.args.AutoCon:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if self.args.model == 'DeepTime':
                        outputs, repr = self.model(batch_x, batch_x_mark, batch_y_mark)
                    else:
                        outputs, repr, pos = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                MSE_loss = F.mse_loss(outputs, batch_y, reduction='none')
                # MSE_loss = F.mse_loss(outputs, batch_x, reduction='none')
                # print(MSE_loss)

                # features = F.normalize(repr, dim=-1)  # B, T, C
                # Fre_local_loss, Fre_global_loss = self.FreLoss(outputs, batch_y)
                # FreLoss = self.FreLoss(outputs, batch_y)
                # print(FreLoss)
                global_pos_labels = timeindex.long()
                # local_loss, global_loss = self.AutoCon_loss(outputs, global_pos_labels)
                local_loss, global_loss = self.FreLoss(repr, pos, global_pos_labels)

                # if self.args.model == 'AutoConCI':
                #     autocon_loss = (local_loss.reshape(B, C, T // 3).mean(dim=2).mean(dim=1) + global_loss.mean(
                #         dim=0)) / 2.0
                # else:
                #     autocon_loss = (local_loss.mean(dim=1) + global_loss) / 2.0
                # print(autocon_loss)
                # exit()
                # loss = MSE_loss.mean() + self.args.AutoCon_lambda * autocon_loss.mean()
                # FreLoss = (local_loss + global_loss) / 2
                FreLoss = (self.args.alpha * local_loss + (1 - self.args.alpha) * global_loss)
                loss = MSE_loss.mean() + self.args.AutoCon_lambda * FreLoss
                # print(loss)
                # exit()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                train_log['loss'].append(loss.item())
                train_log['FreLoss'].append(FreLoss.detach().cpu())
                train_log['MSE_loss'].append(MSE_loss.detach().cpu())

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_log['loss'] = np.average(train_log['loss'])
            # train_log['FreLoss'] = torch.cat(train_log['FreLoss'], dim=0)
            train_log['FreLoss'] = torch.stack(train_log['FreLoss'])

            train_log['MSE_loss'] = torch.cat(train_log['MSE_loss'], dim=0)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} "
                  f"Train Loss: {train_log['loss']:.4f} (Forecasting Loss:{train_log['MSE_loss'].mean():.4f} + "
                  f"Fre Loss:{train_log['FreLoss'].mean():.4f} x Lambda({self.args.AutoCon_lambda})), "
                  f"Vali MSE Loss: {vali_loss:.4f} Test MSE Loss: {test_loss:.4f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                timeindex = timeindex.float().to(self.device)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if not self.args.AutoCon:
                    # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    if self.args.model == 'DeepTime':
                        outputs, repr = self.model(batch_x, batch_x_mark, batch_y_mark)
                    else:
                        outputs, repr = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'DeepTime':
                        outputs, repr = self.model(batch_x, batch_x_mark, batch_y_mark)
                    else:
                        outputs, repr, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # result_scripts save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # dilate_e, shape_e, temporal_e = shape_metric(preds, trues)  # These metrics take a long time to calculate.
        dilate_e, shape_e, temporal_e = 0.0, 0.0, 0.0

        print(
            f'mse:{mse}, mae:{mae}, mape:{mape}, mspe:{mspe} dilate:{dilate_e:.7f}, Shapedtw:{shape_e:.7f}, Temporaldtw:{temporal_e:.7f}')
        f = open("60_result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        fileName = "result_scripts/best_result_" + str(self.args.model) + "_" + str(self.args.data_path) + "_" + \
                   str(self.args.seq_len) + "_" + str(self.args.label_len) + "_" + str(self.args.pred_len) + ".txt"

        isSave = False
        if os.path.exists('./' + fileName):
            f = open(fileName, 'r')

            old_mse = f.readline().strip()
            old_mae = f.readline().strip()

            old_mse = float(old_mse)
            if mse < old_mse:
                isSave = True

            f.close()
        else:
            self.save_best_result(fileName, 'a', mse, mae)

        if isSave:
            self.save_best_result(fileName, 'w', mse, mae)

        if self.args.save:
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            print(preds.shape)
            print(trues.shape)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return mse, mae, mape, mspe, dilate_e, shape_e, temporal_e

    def save_best_result(self, fileName, t, mse, mae):
        print('this is the best result_scripts!!!')
        f = open(fileName, t)

        f.write(str(mse))
        f.write('\n')
        f.write(str(mae))
        f.write('\n')
        f.write('python -u run.py --is_training 1')
        f.write(' --' + 'AutoCon ' + str(self.args.AutoCon))
        f.write(' --' + 'AutoCon_multiscales ' + str(self.args.AutoCon_multiscales))
        f.write(' --' + 'AutoCon_wnorm ' + str(self.args.AutoCon_wnorm))
        f.write(' --' + 'AutoCon_lambda ' + str(self.args.AutoCon_lambda))
        f.write(' --' + 'Auto_decomp ' + str(self.args.Auto_decomp))
        f.write(' --' + 'd_model ' + str(self.args.d_model))
        f.write(' --' + 'd_ff ' + str(self.args.d_ff))
        f.write(' --' + 'e_layers ' + str(self.args.e_layers))
        f.write(' --' + 'target ' + str(self.args.target))
        f.write(' --' + 'root_path ' + str(self.args.root_path))
        f.write(' --' + 'data_path ' + str(self.args.data_path))
        f.write(' --' + 'model_id ' + str(self.args.model_id))
        f.write(' --' + 'model ' + str(self.args.model))
        f.write(' --' + 'data ' + str(self.args.data))
        f.write(' --' + 'seq_len ' + str(self.args.seq_len))
        f.write(' --' + 'label_len ' + str(self.args.label_len))
        f.write(' --' + 'pred_len ' + str(self.args.pred_len))
        f.write(' --' + 'enc_in ' + str(self.args.enc_in))
        f.write(' --' + 'c_out ' + str(self.args.c_out))
        f.write(' --' + 'des ' + str(self.args.des))
        f.write(' --' + 'itr ' + str(self.args.itr))
        f.write(' --' + 'batch_size ' + str(self.args.batch_size))
        f.write(' --' + 'learning_rate ' + str(self.args.learning_rate))
        f.write(' --' + 'features ' + str(self.args.features))
        f.write(' --' + 'train_epochs ' + str(self.args.train_epochs))
        f.write(' --' + 'patience ' + str(self.args.patience))
        f.write(' --' + 'global_k ' + str(self.args.global_k))
        f.write(' --' + 'local_k ' + str(self.args.local_k))
        f.write(' --' + 'pos_k ' + str(self.args.pos_k))
        f.write(' --' + 'acf_k ' + str(self.args.acf_k))
        f.write(' --' + 'top_k_decomp ' + str(self.args.top_k_decomp))
        f.write(' --' + 'top_k_multi ' + str(self.args.top_k_multi))
        f.write(' --' + 'alpha ' + str(self.args.alpha))
        f.close()
