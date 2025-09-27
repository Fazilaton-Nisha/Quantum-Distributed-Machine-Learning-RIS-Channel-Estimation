# updated runner file with loss curve
r'''
    This is training runner for Hierarchical Deep Channel Estimation (HDCE).
'''

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from Estimators_QuantumNAT_onchipQNN import NMSELoss, SC_P128, Conv_P128, FC_P128, QSC_P128
from generate_data import DatasetFolder_DML


class Y2HRunner():
    def __init__(self):
        self.Pilot_num = 128
        self.data_len = 20000
        self.SNRdb = 10
        self.num_workers = 0
        self.batch_size = 256
        self.batch_size_DML = 256
        self.lr = 1e-3
        self.lr_decay = 30
        self.lr_threshold = 1e-6
        self.n_epochs = 100
        self.print_freq = 50
        self.optimizer = 'adam'
        self.train_test_ratio = 0.9

        # Initialize variables to store training history
        self.train_QSC_losses = []
        self.val_QSC_losses = []  # New: Store validation losses
        self.val_QSC_accuracies = []  # New: Store validation accuracies

    def get_optimizer(self, parameters, lr):
        if self.optimizer == 'adam':
            return optim.Adam(parameters, lr=lr)
        elif self.optimizer == 'sgd':
            return optim.SGD(parameters, lr=lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.optimizer))

    def get_data(self, data_len, indicator, uid):
        Yp = np.load(
            'available_data/Yp' + str(indicator) + '_' + str(self.Pilot_num) + '_1024_' + str(self.SNRdb) + 'dB_' + str(
                uid) + '_datalen_' + str(data_len) + '.npy')
        Hlabel = np.load('available_data/Hlabel' + str(indicator) + '_' + str(self.Pilot_num) + '_1024_' + str(
            self.SNRdb) + 'dB_' + str(uid) + '_datalen_' + str(data_len) + '.npy')
        Hperf = np.load('available_data/Hperf' + str(indicator) + '_' + str(self.Pilot_num) + '_1024_' + str(
            self.SNRdb) + 'dB_' + str(uid) + '_datalen_' + str(data_len) + '.npy')

        print('data loaded for scenario' + str(indicator) + ' user' + str(uid) + '!')
        Indicator = []
        for i in range(data_len):
            Indicator.append(indicator)
        Indicator = np.stack(Indicator, axis=0)

        Yp = Yp[:data_len]
        Hlabel = Hlabel[:data_len]
        Hperf = Hperf[:data_len]

        start = int(Yp.shape[0] * self.train_test_ratio)
        Yp_train, Yp_val = Yp[:start], Yp[start:]
        Hlabel_train, Hlabel_val = Hlabel[:start], Hlabel[start:]
        Hperf_train, Hperf_val = Hperf[:start], Hperf[start:]
        Indicator_train, Indicator_val = Indicator[:start], Indicator[start:]

        return [Yp_train, Hlabel_train, Hperf_train, Indicator_train], [Yp_val, Hlabel_val, Hperf_val, Indicator_val]

    def get_dataloader_DML(self, data_len):
        td00, vd00 = self.get_data(data_len, 0, 0)
        td01, vd01 = self.get_data(data_len, 0, 1)
        td02, vd02 = self.get_data(data_len, 0, 2)
        td10, vd10 = self.get_data(data_len, 1, 0)
        td11, vd11 = self.get_data(data_len, 1, 1)
        td12, vd12 = self.get_data(data_len, 1, 2)
        td20, vd20 = self.get_data(data_len, 2, 0)
        td21, vd21 = self.get_data(data_len, 2, 1)
        td22, vd22 = self.get_data(data_len, 2, 2)

        # dataLoader for training or val
        train_dataset = DatasetFolder_DML(td00, td01, td02, td10, td11, td12, td20, td21, td22)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size_DML, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        val_dataset = DatasetFolder_DML(vd00, vd01, vd02, vd10, vd11, vd12, vd20, vd21, vd22)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size_DML, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        return train_loader, val_loader

    def get_HDCE_loss(self, td, Conv, CE, criterion, device):
        Yp = td[0]
        Hlabel = td[1]
        Hperfect = td[2]
        bs = len(Yp)

        # complex--->real
        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float().to(device)
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float().to(device)

        # the input and output
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(device)
        h_out = Conv(Yp_input)
        Hhat = CE(h_out)

        loss = criterion(Hhat, label_out)
        loss_perf = criterion(Hhat, perfect_out)

        return loss, loss_perf

    def get_HDCE_estimate(self, vd, Conv, CE, device):
        Yp = vd[0]
        Hlabel = vd[1]
        Hperfect = vd[2]
        bs = len(Yp)

        # complex--->real
        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float().to(device)
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float().to(device)

        # the input and output
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(device)
        h_out = Conv(Yp_input)
        Hhat = CE(h_out)

        return Hhat, label_out, perfect_out

    def train_Conv_Linear_of_HDCE(self):
        gpu_list = '0,1,2,3'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        device = 'cuda'

        Conv0 = Conv_P128()
        Conv1 = Conv_P128()
        Conv2 = Conv_P128()
        CE = FC_P128()

        if len(gpu_list.split(',')) > 1:
            Conv0 = torch.nn.DataParallel(Conv0).to(device)
            Conv1 = torch.nn.DataParallel(Conv1).to(device)
            Conv2 = torch.nn.DataParallel(Conv2).to(device)
            CE = torch.nn.DataParallel(CE).to(device)
        else:
            Conv0.to(device)
            Conv1.to(device)
            Conv2.to(device)
            CE.to(device)

        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded!')

        criterion = NMSELoss()
        optimizer_Conv0 = self.get_optimizer(Conv0.parameters(), self.lr)
        optimizer_Conv1 = self.get_optimizer(Conv1.parameters(), self.lr)
        optimizer_Conv2 = self.get_optimizer(Conv2.parameters(), self.lr)
        optimizer_CE = self.get_optimizer(CE.parameters(), self.lr)

        best_nmse = 1000.

        print('Everything prepared well, start to train HDCE Conv+Linear...')

        for epoch in range(self.n_epochs):
            current_lr = optimizer_Conv0.param_groups[0]['lr']
            print(
                'HDCE Conv+Linear:' f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            Conv0.train()
            Conv1.train()
            Conv2.train()
            CE.train()
            Conv = [Conv0, Conv1, Conv2]

            for it, (td00, td01, td02, td10, td11, td12, td20, td21, td22) in enumerate(train_loader):
                sutd = [[td00, td01, td02], [td10, td11, td12], [td20, td21, td22]]

                optimizer_Conv0.zero_grad()
                optimizer_Conv1.zero_grad()
                optimizer_Conv2.zero_grad()
                optimizer_CE.zero_grad()

                total_loss = 0
                total_loss_perf = 0

                for sid in range(3):
                    for uid in range(3):
                        loss, loss_perf = self.get_HDCE_loss(sutd[sid][uid], Conv[sid], CE, criterion, device)
                        loss = loss / 9
                        loss_perf = loss_perf / 9
                        total_loss += loss
                        total_loss_perf += loss_perf
                        loss.backward()

                optimizer_Conv0.step()
                optimizer_Conv1.step()
                optimizer_Conv2.step()
                optimizer_CE.step()

                if it % self.print_freq == 0:
                    print(
                        f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {total_loss.item():.5f}\t Loss_perf {total_loss_perf.item():.5f}')

            Conv0.eval()
            Conv1.eval()
            Conv2.eval()
            CE.eval()
            Conv = [Conv0, Conv1, Conv2]

            with torch.no_grad():
                Hhat_list = []
                Hlabel_list = []
                Hperfect_list = []

                for vd00, vd01, vd02, vd10, vd11, vd12, vd20, vd21, vd22 in val_loader:
                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]

                    for sid in range(3):
                        for uid in range(3):
                            Hhat, label_out, perfect_out = self.get_HDCE_estimate(suvd[sid][uid], Conv[sid], CE, device)
                            Hhat_list.append(Hhat)
                            Hlabel_list.append(label_out)
                            Hperfect_list.append(perfect_out)

                Hhat = torch.cat(Hhat_list, dim=0)
                Hlable = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)
                nmse = criterion(Hhat, Hlable)
                nmse_perf = criterion(Hhat, Hperfect)

                if epoch == self.n_epochs - 1:
                    torch.save({'conv': Conv0.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv0_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'conv': Conv1.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv1_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'conv': Conv2.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv2_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'linear': CE.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Linear_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    print('HDCE finally saved!')

                if nmse < best_nmse:
                    torch.save({'conv': Conv0.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv0_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'conv': Conv1.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv1_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'conv': Conv2.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv2_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'linear': CE.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Linear_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    best_nmse = nmse.item()
                    print('HDCE saved!')

                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || NMSE {nmse.item():.5f}, NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')
                print('==============================================================')

            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer_Conv0.param_groups[0]['lr'] = optimizer_Conv0.param_groups[0]['lr'] * 0.5
                    optimizer_Conv1.param_groups[0]['lr'] = optimizer_Conv1.param_groups[0]['lr'] * 0.5
                    optimizer_Conv2.param_groups[0]['lr'] = optimizer_Conv2.param_groups[0]['lr'] * 0.5
                    optimizer_CE.param_groups[0]['lr'] = optimizer_CE.param_groups[0]['lr'] * 0.5

                if optimizer_Conv0.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer_Conv0.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_Conv1.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_Conv2.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_CE.param_groups[0]['lr'] = self.lr_threshold

    def get_SE_loss(self, td, CNN, device):
        Yp = td[0]
        indicator = td[3]
        bs = len(Yp)
        label_out = indicator.long().to(device)
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
        pred_indicator = CNN(Yp_input.reshape(bs, 2, 16, 8))
        loss = F.nll_loss(pred_indicator, label_out)
        return loss

    def get_SE_estimate(self, td, CNN, device):
        Yp = td[0]
        indicator = td[3]
        bs = len(Yp)
        label_out = indicator.long().to(device)
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
        pred_indicator = CNN(Yp_input.reshape(bs, 2, 16, 8))
        return pred_indicator, label_out




    def train_QSC_P128(self):
        gpu_list = '0,1,2,3'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        device = 'cuda'

        # CHANGE 1: Initialize optimized model with conservative settings
        model = QSC_P128(
            use_quantumnat=False,  # DISABLE QuantumNAT
            use_gradient_pruning=False  # DISABLE gradient pruning
        ).to(device)

        # CHANGE 2: Slightly lower learning rate for stability
        #optimizer = torch.optim.Adam(model.parameters(), lr=self.lr * 0.9)  # Reduced by 10%
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        best_acc = 0

        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded for QSC with QuantumNAT + On-chip QNN optimization!')

        # Clear previous training history
        self.train_QSC_losses = []

        for epoch in range(self.n_epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0

            for it, (td00, td01, td02, td10, td11, td12, td20, td21, td22) in enumerate(train_loader):
                sutd = [[td00, td01, td02], [td10, td11, td12], [td20, td21, td22]]

                optimizer.zero_grad()
                loss = 0

                for sid in range(3):
                    for uid in range(3):
                        data_batch = sutd[sid][uid]
                        inputs = torch.cat([data_batch[0].real, data_batch[0].imag], dim=1).float().to(device)
                        inputs = inputs.reshape(inputs.size(0), 2, 16, 8)
                        labels = data_batch[3].long().to(device)

                        # CRITICAL FIX: Ensure labels are 1D
                        while len(labels.shape) > 1:
                            labels = labels.squeeze(-1)

                        outputs = model(inputs)

                        # CRITICAL FIX: Ensure outputs are 2D
                        if len(outputs.shape) != 2:
                            raise ValueError(
                                f"Model outputs should be 2D (batch_size, num_classes), got shape {outputs.shape}")

                        # CRITICAL FIX: Ensure labels are in valid range
                        labels = torch.clamp(labels, 0, outputs.size(-1) - 1)

                        loss += F.nll_loss(outputs, labels) / 9

                loss.backward()

                # CHANGE 3: Apply gradient pruning after backward()
                model.apply_gradient_pruning()  # On-chip QNN optimization

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            self.train_QSC_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{self.n_epochs}, Average Loss: {avg_epoch_loss:.4f}")

            # Validation (NO CHANGES NEEDED)
            model.eval()
            correct, total = 0, 0
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for vd00, vd01, vd02, vd10, vd11, vd12, vd20, vd21, vd22 in val_loader:
                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]
                    batch_val_loss = 0
                    for sid in range(3):
                        for uid in range(3):
                            data_batch = suvd[sid][uid]
                            inputs = torch.cat([data_batch[0].real, data_batch[0].imag], dim=1).float().to(device)
                            inputs = inputs.reshape(inputs.size(0), 2, 16, 8)
                            labels = data_batch[3].long().to(device)

                            # Fix labels shape
                            while len(labels.shape) > 1:
                                labels = labels.squeeze(-1)

                            outputs = model(inputs)
                            # Calculate validation loss
                            batch_val_loss += F.nll_loss(outputs, labels) / 9
                            pred = outputs.argmax(dim=1)
                            correct += pred.eq(labels).sum().item()
                            total += labels.size(0)

                    val_loss += batch_val_loss.item()
                    val_batches += 1

            # Calculate validation metrics
            avg_val_loss = val_loss / val_batches
            acc = correct / total
            # Store validation metrics
            self.val_QSC_losses.append(avg_val_loss)
            self.val_QSC_accuracies.append(acc)

            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {acc:.2%}")

            # CHANGE 4: Update save filenames to indicate optimization
            if acc > best_acc:
                torch.save(model.state_dict(), os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                                            f'QSC_OPT_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))  # Added _OPT
                best_acc = acc
                print("Optimized QML saved (best so far).")

            if epoch == self.n_epochs - 1:
                torch.save(model.state_dict(), os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                                            f'QSC_OPT_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))  # Added _OPT
                print('Optimized QSC finally saved!')





if __name__ == '__main__':
    runner = Y2HRunner()

    # # Train the Quantum Scenario Classifier
    print("=== Training Quantum Scenario Classifier ===")
    start_qsc = time.time()
    runner.train_QSC_P128()
    end_qsc = time.time()
    print(f"Quantum Scenario Classifier training time: {(end_qsc - start_qsc) / 60:.2f} minutes")



    print("HDCE and QML training completed!")














