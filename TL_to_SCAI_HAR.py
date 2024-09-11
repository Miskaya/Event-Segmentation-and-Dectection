import os
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import pickle
import copy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import FEN, FLN
import time
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Set a fixed seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Create datasets for time series analysis
def create_dataset(X, y, time_steps, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        x = X.iloc[i : (i + time_steps)].values
        labels = y.iloc[i : i + time_steps]
        Xs.append(x)
        ys.append(stats.mode(labels)[0])
    Xs = np.swapaxes(Xs, 1, 2)
    return np.array(Xs), np.array(ys).reshape(-1, 1)

# Main training loop
def training_loop(train_loader, val_loader, fen_model, fln_model, epochs, device, fen_lr, fln_lr, patience):
    fen_optimizer = optim.Adam(fen_model.parameters(), lr=fen_lr)
    fln_optimizer = optim.Adam(fln_model.parameters(), lr=fln_lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    best_acc = 0.0
    no_improvement_count = 0
    
    for epoch in range(epochs):
        # Training
        fen_model.train()
        fln_model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            fen_optimizer.zero_grad()
            fln_optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            
            fen_outputs = []
            for signal in range(inputs.size(1)):
                signal_input = inputs[:, signal, :].unsqueeze(1)
                fen_output = fen_model(signal_input)
                fen_outputs.append(fen_output)
            fen_outputs_combined = torch.cat(fen_outputs, dim=2)
            fln_output = fln_model(fen_outputs_combined)
            if len(fln_output.shape) == 1:
                fln_output = fln_output.unsqueeze(0)
            loss = criterion(fln_output, labels.view(-1))
            
            loss.backward()
            fen_optimizer.step()
            fln_optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(fln_output, 1)
            correct_train += (predicted == labels.view(-1)).sum().item()
            total_train += labels.size(0)
            
        train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation
        fen_model.eval()
        fln_model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                fen_outputs = []
                for signal in range(inputs.size(1)):
                    signal_input = inputs[:, signal, :].unsqueeze(1)
                    fen_output = fen_model(signal_input)
                    fen_outputs.append(fen_output)
                fen_outputs_combined = torch.cat(fen_outputs, dim=2)
                fln_output = fln_model(fen_outputs_combined)
                if len(fln_output.shape) == 1:
                    fln_output = fln_output.unsqueeze(0)
                loss = criterion(fln_output, labels.view(-1))
                total_val_loss += loss.item()
                _, predicted = torch.max(fln_output, 1)
                correct_val += (predicted == labels.view(-1)).sum().item()
                total_val += labels.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        print(f'Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Check for early stopping
        if val_acc > best_acc:
            best_loss = val_loss
            best_acc = val_acc
            no_improvement_count = 0
            best_fen_model_state = copy.deepcopy(fen_model.state_dict())
            best_fln_model_state = copy.deepcopy(fln_model.state_dict())
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Best Validation Loss: {best_loss:.4f} with Accuracy: {best_acc:.4f}%")

    return fen_model, fln_model, best_fen_model_state, best_fln_model_state, best_acc, best_loss

# Evaluate the trained models on test data
def evaluate_model(test_loader, fen_model, fln_model, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_labels = []
    all_predictions = []
    
    class_correct = np.zeros(11)
    class_total = np.zeros(11)
    
    fen_model.eval()
    fln_model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            fen_outputs = []
            for signal in range(inputs.size(1)):
                signal_input = inputs[:, signal, :].unsqueeze(1)
                fen_output = fen_model(signal_input)
                fen_outputs.append(fen_output)
            fen_outputs_combined = torch.cat(fen_outputs, dim=2)
            fln_output = fln_model(fen_outputs_combined)
            if len(fln_output.shape) == 1:
                fln_output = fln_output.unsqueeze(0)
            loss = criterion(fln_output, labels.view(-1))
            total_loss += loss.item()

            _, predicted = torch.max(fln_output, 1)
            total_correct += (predicted == labels.view(-1)).sum().item()
            total_samples += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Calculate per class accuracy
            correct_tensor = predicted == labels.view(-1)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += correct_tensor[i].item()
                class_total[label] += 1

    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * total_correct / total_samples
    
    # Calculate weighted metrics
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted')

    # Calculate class-specific accuracies
    for i in range(11):
        if class_total[i] > 0:
            print(f'Accuracy of label {i} : {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'Accuracy of label {i} : N/A - No samples')
        
    return test_loss, test_accuracy, f1, precision, recall

# Standardization of time-series data before training
def standardize_data_fine_tuning(X_train, X_val, X_test):
    # Flatten the 3D arrays to 2D for standardization
    nsamples, nfeatures, ntimesteps = X_train.shape
    X_train_flat = X_train.reshape((nsamples * ntimesteps, nfeatures))
    X_val_flat = X_val.reshape((X_val.shape[0] * X_val.shape[2], X_val.shape[1]))
    X_test_flat = X_test.reshape((X_test.shape[0] * X_test.shape[2], X_test.shape[1]))

    # Standardize the flattened data
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)

    # Reshape back to the original 3D shape
    X_train_std = X_train_flat.reshape((nsamples, nfeatures, ntimesteps))
    X_val_std = X_val_flat.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    X_test_std = X_test_flat.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    
    return X_train_std, X_val_std, X_test_std   

# Running the entire experimental setup           
def run_experiment(config, data_splits, split_ind):
    print(f"split_index: {split_ind}, Running experiment with config: {config}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    results_summary = {}
    root_dir = '../segmented_data/'
    pkl_files = [file for file in os.listdir(root_dir) if file.endswith('Pre_processed_SCAI_HAR.pkl')]
    chosen_file = pkl_files[0]
    setup_seed(1)
    print(f"Processing file: {chosen_file}")

    with open(os.path.join(root_dir, chosen_file), 'rb') as f:
        data = pickle.load(f)
    dataset_name = os.path.splitext(chosen_file)[0]

    print(f"dataset:", dataset_name)
    
    split_index = split_ind
    train_subjects, val_subjects, test1_subjects, test2_subjects = data_splits['modified_combined_sensor_data'][split_index]
    train_subjects = np.concatenate((train_subjects, test1_subjects))
    print(train_subjects, val_subjects, test2_subjects)
    
    # Filter data by removing certain labels, create a copy to avoid warnings
    filtered_data = data[~data['label'].isin([4, 5, 7, 9])].copy()

    # Creating label mapping
    label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 8: 5, 10: 6, 11: 7}
    filtered_data.loc[:, 'label'] = filtered_data['label'].map(label_mapping)

    # Splitting data by subject
    train_data = filtered_data[filtered_data['subject'].isin(train_subjects)]
    val_data = filtered_data[filtered_data['subject'].isin(val_subjects)]
    test_data = filtered_data[filtered_data['subject'].isin(test2_subjects)]

    time_steps = 400
    step = 200

    # Create dataset instances for training, validation, and testing
    X_train, y_train = create_dataset(train_data.drop(columns=['label', 'subject']), train_data['label'], time_steps, step)
    X_val, y_val = create_dataset(val_data.drop(columns=['label', 'subject']), val_data['label'], time_steps, step)
    X_test, y_test = create_dataset(test_data.drop(columns=['label', 'subject']), test_data['label'], time_steps, step)

    # Standardize the data
    X_train_std, X_val_std, X_test_std = standardize_data_fine_tuning(X_train, X_val, X_test)
    
    # Load data into DataLoader
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_std, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_std, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=False, drop_last=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test_std, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=False, drop_last=False)

    unique_labels = np.unique(y_train)
    num_classes = len(unique_labels)
    num_signals = X_train.shape[1]
        
    # Configuration parameters for the models
    out_channels1 = config['out_channels1']
    out_channels2 = config['out_channels2']
    out_channels3 = config['out_channels3']
    out_channels4 = config['out_channels4']
    lstm_out_channels = config['lstm_out_channels']
    epochs = config['epoch']
    fen_lr = config['fen_lr']
    fln_lr = config['fln_lr']
    
    # initialize models
    fen_model = FEN(1, out_channels1, out_channels2, out_channels3, out_channels4).to(device)
    fln_model = FLN(out_channels4*num_signals, lstm_out_channels, num_classes).to(device)
    
    weights_folder = 'weights_0'
    fen_weights_file = os.path.join(weights_folder, f'CNNBiLSTMAttn_fen_weights_{out_channels1}_{out_channels2}_{out_channels3}_{out_channels4}_{lstm_out_channels}.pth')
    print(fen_weights_file)
    if os.path.exists(fen_weights_file):
        fen_model.load_state_dict(torch.load(fen_weights_file))
        print("FEN weights loaded")
    for param in fen_model.parameters():
        param.requires_grad = True
    
    fen_model, fln_model, best_fen_model_state, best_fln_model_state, best_acc, best_loss = training_loop(train_loader, val_loader, fen_model, fln_model, epochs, device, fen_lr, fln_lr, 10)
    
    weights_dir = '../sensei_v2_pickle/weights_after_tf'
    # Ensure the directory exists    
    os.makedirs(weights_dir, exist_ok=True)
    fen_weights_path = os.path.join(weights_dir, 'best_fen_weights_0.pth')
    # Save the best FEN model weights
    torch.save(best_fen_model_state, fen_weights_path)
    print(f"Best FEN model weights saved to: {fen_weights_path}")
    fln_weights_path = os.path.join(weights_dir, 'best_fln_weights_0.pth')
    torch.save(best_fln_model_state, fln_weights_path)
    print(f"Best FLN model weights saved to: {fln_weights_path}")    
    
    if best_fen_model_state is not None:
        if os.path.exists(fen_weights_path) and os.path.exists(fln_weights_path):
            # Load saved weights for FEN and FLN models
            fen_model.load_state_dict(torch.load(fen_weights_path))
            fln_model.load_state_dict(torch.load(fln_weights_path))
            print("Loaded FEN and FLN model weights from saved files.")
        
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(test_loader, fen_model, fln_model, device)
        
        results_summary[dataset_name] = {
        "Best Validation Loss": best_loss,
        "Best Validation Accuracy": best_acc,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "F1-Score": test_f1,
        "Precision": test_precision,
        "Recall": test_recall
        }
                
        print(f'Test Loss for {dataset_name}: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, '
            f'F1-Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')          
    return results_summary

def pretty_print_results(results, split_index):
    print(f"\nResults for split index: {split_index}")
    for config, datasets_results in results.items():
        print(f"Configuration: {config}")
        for dataset_name, metrics in datasets_results.items():
            print(f"  Dataset: {dataset_name}")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        print("\n")
        
start_time = time.time() 
parameter_grid = {
        'out_channels1': [128],
        'out_channels2': [256],
        'out_channels3': [256],
        'out_channels4': [512],
        'lstm_out_channels': [128],
        'fen_lr': [0.001],
        'fln_lr': [0.0001],
        'epoch': [100],
}
all_results = {}
keys, values = zip(*parameter_grid.items())

with open('data_splits_independent_dataset.pkl', 'rb') as f:
    data_splits = pickle.load(f)

for combination in itertools.product(*values):
    config = dict(zip(keys, combination))
    for split_index in range(1):
        final_results = run_experiment(config,data_splits, split_index)
        all_results[str(config)] = final_results
        pretty_print_results(all_results, split_index)
    
    
end_time = time.time() 
total_time_seconds = end_time - start_time
hours = int(total_time_seconds // 3600)
minutes = int((total_time_seconds % 3600) // 60) 
print(f"\nTotal Execution Time: {hours} hours and {minutes} minutes")