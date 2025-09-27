import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from Estimators import NMSELoss, DCE_P128, SC_P128, Conv_P128, FC_P128
from Runner_P128 import OptimizedQSC_P128  # Import your quantum classifier
from generate_data import generate_MMSE_estimate, generate_datapair, DatasetFolder
import matplotlib.pyplot as plt


class model_val():
    def __init__(self):
        super().__init__()
        self.training_SNRdb = 10
        self.num_workers = 8
        self.batch_size = 200
        self.batch_size_DML = 256
        self.training_data_len = 20000
        self.indicator = -1
        self.data_len_for_test = 10000
        self.Pilot_num = 128

    def load_model_state_dict(self, model, filepath, fallback_key=None):
        """
        Load model state dict with automatic handling of DataParallel key mismatches
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load the checkpoint
        checkpoint = torch.load(filepath, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and fallback_key and fallback_key in checkpoint:
            state_dict = checkpoint[fallback_key]
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle DataParallel key mismatch
        if list(state_dict.keys())[0].startswith('module.') and not hasattr(model, 'module'):
            # Model was saved with DataParallel but current model doesn't have it
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key[7:]  # Remove 'module.' prefix
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        elif not list(state_dict.keys())[0].startswith('module.') and hasattr(model, 'module'):
            # Model was saved without DataParallel but current model has it
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = 'module.' + key
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        try:
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model from {filepath}")
        except RuntimeError as e:
            print(f"Error loading model from {filepath}: {e}")
            raise

    def test_for_CE_P128_for_all_scenarios(self):
        Pilot_num = 128
        SNRdb = np.arange(5, 16, 2)

        # Load classical scenario classifier
        SC_classical = SC_P128()
        SC_classical = torch.nn.DataParallel(SC_classical).to(device)
        fp_classical = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                                    f'{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML_SC.pth')
        self.load_model_state_dict(SC_classical, fp_classical, fallback_key='cnn')

        # Load quantum scenario classifier
        # Load quantum scenario classifier
        SC_quantum = OptimizedQSC_P128
        SC_quantum = torch.nn.DataParallel(SC_quantum).to(device)
        fp_quantum = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                                  f'QSC_optimized_best.pth')  # match Runner's save path
        try:
            # Pass the correct fallback key
            self.load_model_state_dict(SC_quantum, fp_quantum, fallback_key='model_state_dict')
        except:
            print("Quantum SC model not found, using classical only")
            SC_quantum = None

        # Load HDCE components
        Conv0 = Conv_P128()
        Conv1 = Conv_P128()
        Conv2 = Conv_P128()
        CE = FC_P128()

        Conv0 = torch.nn.DataParallel(Conv0).to(device)
        Conv1 = torch.nn.DataParallel(Conv1).to(device)
        Conv2 = torch.nn.DataParallel(Conv2).to(device)
        CE = torch.nn.DataParallel(CE).to(device)

        # Load Conv and CE models
        for name, model, key in [
            ("Conv0", Conv0, "conv"),
            ("Conv1", Conv1, "conv"),
            ("Conv2", Conv2, "conv"),
            ("Linear", CE, "linear")]:
            fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                              f'{name}_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
            self.load_model_state_dict(model, fp, fallback_key=key)

        # Set models to evaluation mode
        SC_classical.eval()
        if SC_quantum:
            SC_quantum.eval()
        Conv0.eval();
        Conv1.eval();
        Conv2.eval();
        CE.eval()

        criterion = NMSELoss()

        # Results storage
        NMSE_for_LS, NMSE_for_MMSE, NMSE_for_HDCE_classical, NMSE_for_HDCE_quantum = [], [], [], []
        SC_accuracy_classical, SC_accuracy_quantum = [], []

        with torch.no_grad():
            for snr in SNRdb:
                print(f'Generating test data for SNR: {snr} dB')
                td = generate_datapair(Ns=self.data_len_for_test, Pilot_num=128, index=self.indicator,
                                       SNRdb=snr, start=self.training_data_len * 3,
                                       training_data_len=self.training_data_len)
                test_dataset = DatasetFolder(td)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                          num_workers=self.num_workers, pin_memory=True,
                                                          drop_last=False)

                Hperfect_list, HLS_list, HMMSE_list = [], [], []
                Hhat_list_classical, Hperfect_list_classical = [], []
                Hhat_list_quantum, Hperfect_list_quantum = [], []
                true_labels, pred_labels_classical, pred_labels_quantum = [], [], []

                for Yp, HLS, Hperfect, indicator in test_loader:
                    # Store true labels
                    true_labels.extend(indicator.numpy())

                    # Generate MMSE estimate
                    HMMSE = generate_MMSE_estimate(HLS.numpy(), sigma2=10 ** (-snr / 10))
                    HMMSE = torch.from_numpy(HMMSE)
                    HMMSE_list.append(torch.cat([HMMSE.real, HMMSE.imag], dim=1).float())

                    bs = Yp.shape[0]
                    label_out = torch.cat([HLS.real, HLS.imag], dim=1).float()
                    perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float()
                    Yp_input = torch.stack([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(device)

                    HLS_list.append(label_out)
                    Hperfect_list.append(perfect_out)

                    # Classical scenario classification
                    pred_classical = SC_classical(Yp_input).argmax(dim=1).cpu().numpy()
                    pred_labels_classical.extend(pred_classical)

                    # Quantum scenario classification (if available)
                    if SC_quantum:
                        pred_quantum = SC_quantum(Yp_input).argmax(dim=1).cpu().numpy()
                        pred_labels_quantum.extend(pred_quantum)

                    # Process with classical SC
                    Yp_class_classical = [[], [], []]
                    label_class_classical = [[], [], []]
                    for i, m in enumerate(pred_classical):
                        Yp_class_classical[m].append(Yp_input[i])
                        label_class_classical[m].append(perfect_out[i])

                    # Process with quantum SC
                    if SC_quantum:
                        Yp_class_quantum = [[], [], []]
                        label_class_quantum = [[], [], []]
                        for i, m in enumerate(pred_quantum):
                            Yp_class_quantum[m].append(Yp_input[i])
                            label_class_quantum[m].append(perfect_out[i])

                    # Process each scenario for classical
                    for scenario in range(3):
                        if len(Yp_class_classical[scenario]):
                            hh = torch.stack(label_class_classical[scenario], dim=0)
                            Hperfect_list_classical.append(hh)
                            yy = torch.stack(Yp_class_classical[scenario], dim=0)

                            if scenario == 0:
                                h_out = Conv0(yy)
                            elif scenario == 1:
                                h_out = Conv1(yy)
                            else:
                                h_out = Conv2(yy)

                            h_out = CE(h_out).cpu()
                            Hhat_list_classical.append(h_out)

                    # Process each scenario for quantum
                    if SC_quantum:
                        for scenario in range(3):
                            if len(Yp_class_quantum[scenario]):
                                hh = torch.stack(label_class_quantum[scenario], dim=0)
                                Hperfect_list_quantum.append(hh)
                                yy = torch.stack(Yp_class_quantum[scenario], dim=0)

                                if scenario == 0:
                                    h_out = Conv0(yy)
                                elif scenario == 1:
                                    h_out = Conv1(yy)
                                else:
                                    h_out = Conv2(yy)

                                h_out = CE(h_out).cpu()
                                Hhat_list_quantum.append(h_out)

                # Calculate NMSE for all methods
                Hperfect = torch.cat(Hperfect_list, dim=0)
                HLS = torch.cat(HLS_list, dim=0)
                HMMSE = torch.cat(HMMSE_list, dim=0)

                nmse_LS = criterion(HLS, Hperfect)
                nmse_MMSE = criterion(HMMSE, Hperfect)

                NMSE_for_LS.append(nmse_LS.item())
                NMSE_for_MMSE.append(nmse_MMSE.item())

                # Classical HDCE
                if len(Hhat_list_classical) > 0:
                    Hhat_classical = torch.cat(Hhat_list_classical, dim=0)
                    Hperfect_classical = torch.cat(Hperfect_list_classical, dim=0)
                    nmse_classical = criterion(Hhat_classical, Hperfect_classical)
                    NMSE_for_HDCE_classical.append(nmse_classical.item())
                else:
                    NMSE_for_HDCE_classical.append(float('nan'))

                # Quantum HDCE
                if SC_quantum and len(Hhat_list_quantum) > 0:
                    Hhat_quantum = torch.cat(Hhat_list_quantum, dim=0)
                    Hperfect_quantum = torch.cat(Hperfect_list_quantum, dim=0)
                    nmse_quantum = criterion(Hhat_quantum, Hperfect_quantum)
                    NMSE_for_HDCE_quantum.append(nmse_quantum.item())
                else:
                    NMSE_for_HDCE_quantum.append(float('nan'))

                # Calculate scenario classification accuracy
                true_labels = np.array(true_labels)
                pred_labels_classical = np.array(pred_labels_classical)
                sc_acc_classical = np.mean(true_labels == pred_labels_classical)
                SC_accuracy_classical.append(sc_acc_classical)

                if SC_quantum:
                    pred_labels_quantum = np.array(pred_labels_quantum)
                    sc_acc_quantum = np.mean(true_labels == pred_labels_quantum)
                    SC_accuracy_quantum.append(sc_acc_quantum)
                else:
                    SC_accuracy_quantum.append(float('nan'))

                print(f'SNR {snr}dB Results:')
                print(f'  LS NMSE: {10 * np.log10(nmse_LS.item()):.2f} dB')
                print(f'  MMSE NMSE: {10 * np.log10(nmse_MMSE.item()):.2f} dB')
                print(
                    f'  HDCE (Classical) NMSE: {10 * np.log10(nmse_classical.item()) if len(Hhat_list_classical) > 0 else "N/A":.2f} dB')
                if SC_quantum:
                    print(
                        f'  HDCE (Quantum) NMSE: {10 * np.log10(nmse_quantum.item()) if len(Hhat_list_quantum) > 0 else "N/A":.2f} dB')
                print(f'  SC Accuracy (Classical): {sc_acc_classical:.4f}')
                if SC_quantum:
                    print(f'  SC Accuracy (Quantum): {sc_acc_quantum:.4f}')

        # Create comparison plots
        self.create_comparison_plots(SNRdb, NMSE_for_LS, NMSE_for_MMSE,
                                     NMSE_for_HDCE_classical, NMSE_for_HDCE_quantum,
                                     SC_accuracy_classical, SC_accuracy_quantum)

        return 0

    def create_comparison_plots(self, SNRdb, nmse_ls, nmse_mmse, nmse_classical, nmse_quantum,
                                acc_classical, acc_quantum):
        """Create comparison plots for classical vs quantum performance"""

        # Plot 1: NMSE Comparison
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(SNRdb, 10 * np.log10(nmse_ls), 'k--', linewidth=2, marker='o', markersize=8, label='LS Algorithm')
        plt.plot(SNRdb, 10 * np.log10(nmse_mmse), 'r--', linewidth=2, marker='s', markersize=8, label='MMSE Algorithm')
        plt.plot(SNRdb, 10 * np.log10(nmse_classical), 'bd-', linewidth=3, marker='^', markersize=10,
                 label='HDCE (Classical SC)')

        if any(~np.isnan(nmse_quantum)):
            plt.plot(SNRdb, 10 * np.log10(nmse_quantum), 'go-', linewidth=3, marker='d', markersize=10,
                     label='HDCE (Quantum SC)')

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('NMSE (dB)', fontsize=12)
        plt.title('Channel Estimation Performance Comparison', fontsize=14)
        plt.xticks(SNRdb)
        plt.ylim(-20, 5)

        # Plot 2: Scenario Classification Accuracy Comparison
        plt.subplot(1, 2, 2)
        plt.plot(SNRdb, acc_classical, 'bd-', linewidth=3, marker='^', markersize=10, label='Classical SC')

        if any(~np.isnan(acc_quantum)):
            plt.plot(SNRdb, acc_quantum, 'go-', linewidth=3, marker='d', markersize=10, label='Quantum SC')

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Scenario Classification Accuracy Comparison', fontsize=14)
        plt.xticks(SNRdb)
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig('results/Quantum_vs_Classical_Comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save detailed results
        results = {
            'SNR_dB': SNRdb.tolist(),
            'NMSE_LS_dB': [10 * np.log10(x) for x in nmse_ls],
            'NMSE_MMSE_dB': [10 * np.log10(x) for x in nmse_mmse],
            'NMSE_HDCE_Classical_dB': [10 * np.log10(x) if not np.isnan(x) else float('nan') for x in nmse_classical],
            'NMSE_HDCE_Quantum_dB': [10 * np.log10(x) if not np.isnan(x) else float('nan') for x in nmse_quantum],
            'Accuracy_Classical': acc_classical,
            'Accuracy_Quantum': acc_quantum
        }

        import json
        with open('results/quantum_classical_comparison.json', 'w') as f:
            json.dump(results, f, indent=4)

        print("Results saved to 'results/quantum_classical_comparison.json'")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Starting HDCE with Quantum vs Classical comparison test...')
    test = model_val()
    test.indicator = -1  # Test on all scenarios
    test.test_for_CE_P128_for_all_scenarios()