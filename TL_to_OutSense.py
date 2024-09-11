import os
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import pickle
import copy
import pandas as pd
from model import FEN, FLN
import time
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import sys
import hashlib
from collections import Counter, defaultdict

# Compute class weights
def compute_class_weights(labels):
    class_counts = np.bincount(labels.flatten())
    total_samples = len(labels)
    weights = total_samples / (len(class_counts) * class_counts)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    print("Class weights:")
    for i, weight in enumerate(weight_tensor):
        print(f"Class {i}: Weight {weight:.4f}")

    return weight_tensor

# Create datasets for the segments
class SegmentDataset(Dataset):
    def __init__(self, full_data, segments, scaler):
        self.data = full_data
        self.segments = segments
        self.scaler = scaler
        
    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        start, end = self.segments[idx]
        segment = self.data.iloc[start:end + 1].copy()
        X = segment.drop(['Activity', 'Subject', 'ChangePoint'], axis=1).values
        y = segment['Activity'].mode()[0]
        
        # Flatten for scaling
        X_flat = X.reshape(-1, X.shape[1])
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(X.shape)  # Reshape back to original shape

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).permute(1, 0)  # Rearranging the dimensions

        return X_tensor, torch.tensor(y, dtype=torch.long)

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

# Load the Pickle file containing all the main segment indexes
def load_segment_indices_for_subject(test_subject):
    data = pd.read_pickle('./segmentation_ruptures/claspy_without.pkl')
    if test_subject < len(data):
        segment_indices = data.iloc[test_subject]['predicted_relative_index']
        print('segment_index',segment_indices)
        return segment_indices
    else:
        raise ValueError(f"No data available for test subject {test_subject}")

# Main training loop
def training_loop(train_loader, val_loader, fen_model, fln_model, epochs, device, fen_lr, fln_lr, fen_wd, fln_wd, patience):
    # Get the labels of the training data to calculate the weights
    y_train = np.concatenate([y for _, y in train_loader])
    weights = compute_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    fen_optimizer = optim.Adam(fen_model.parameters(), lr=fen_lr, weight_decay=fen_wd)
    fln_optimizer = optim.Adam(fln_model.parameters(), lr=fln_lr, weight_decay=fln_wd)
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
def evaluate_model(test_loader, fen_model, fln_model, device, time_steps, step):
    fen_model.eval()
    fln_model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    total_segments = 0
    correct_segments = 0

    label_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, true_label in test_loader:
            inputs = inputs.to(device)
            true_label = true_label.to(device)

            segment_predictions = []
            segment_loss = []

            for start in range(0, inputs.size(2) - time_steps + 1, step):
                end = start + time_steps
                window_input = inputs[:, :, start:end]

                fen_outputs = []
                for feature in range(window_input.size(1)):
                    feature_input = window_input[:, feature, :].unsqueeze(1)
                    fen_output = fen_model(feature_input)
                    fen_outputs.append(fen_output)

                fen_outputs_combined = torch.cat(fen_outputs, dim=2)
                fln_output = fln_model(fen_outputs_combined)
                loss = criterion(fln_output, true_label.view(-1))
                segment_loss.append(loss.item())
                _, predicted = torch.max(fln_output, 1)
                segment_predictions.append(predicted.item())
            avg_segment_loss = sum(segment_loss) / len(segment_loss)
            total_loss += avg_segment_loss

            most_common, num_most_common = Counter(segment_predictions).most_common(1)[0]

            all_labels.append(true_label.item())
            all_predictions.append(most_common)

            label_stats[true_label.item()]['total'] += 1
            if most_common == true_label.item():
                correct_segments += 1
                label_stats[true_label.item()]['correct'] += 1
            total_segments += 1

    average_loss = total_loss / total_segments if total_segments > 0 else 0
    accuracy = correct_segments / total_segments if total_segments > 0 else 0
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted')

    print(f"Overall Test Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Total segments: {total_segments}, Correctly Predicted: {correct_segments}")
    for label, stats in label_stats.items():
        if stats['total'] > 0:
            label_accuracy = stats['correct'] / stats['total']
            print(f"GT: {label} - Total: {stats['total']}, Correct: {stats['correct']}, Accuracy: {label_accuracy:.4%}")
        else:
            print(f"GT: {label} - No segments.")
    return average_loss, accuracy, f1, precision, recall

# Standardization of time-series data before training
def standardize_data_fine_tuning(X_train, X_val):
    # Flatten the 3D arrays to 2D for standardization
    scaler = StandardScaler()
    nsamples, nfeatures, ntimesteps = X_train.shape
    X_train_flat = X_train.reshape((nsamples * ntimesteps, nfeatures))
    X_val_flat = X_val.reshape((X_val.shape[0] * X_val.shape[2], X_val.shape[1]))

    # Standardize the flattened data
    X_train_std_flat = scaler.fit_transform(X_train_flat)
    X_val_std_flat = scaler.transform(X_val_flat)

    # Reshape back to the original 3D shape
    X_train_std = X_train_std_flat.reshape((nsamples, nfeatures, ntimesteps))
    X_val_std = X_val_std_flat.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))

    return X_train_std, X_val_std, scaler

# Running the entire experimental setup        
def run_experiment(config, data, test_subject):
    print(f"Test subject: {test_subject}, Running experiment with config: {config}")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
    results_summary = {}
    setup_seed(1)

    '''
    arm raises: 0
    assisted propulsion: 1
    changing clothes: 2
    transfer: 3
    pressure relief: 4
    resting: 5
    self propulsion: 6
    washing hands: 7
    '''

    subject_hash = hashlib.md5(str(test_subject).encode()).hexdigest()
    random_seed = int(subject_hash, 16) % (2**32)
    
    train_val_data = data[data['Subject'] != test_subject]
    test_data = data[data['Subject'] == test_subject]
    train_subjects, val_subjects = train_test_split(train_val_data['Subject'].unique(), test_size=0.25, random_state=random_seed)

    print(train_subjects, val_subjects, test_subject)
    
    train_data = train_val_data[train_val_data['Subject'].isin(train_subjects)]
    val_data = train_val_data[train_val_data['Subject'].isin(val_subjects)]

    time_steps = 400
    step = 200

    X_train, y_train = create_dataset(train_data.drop(columns=['Activity', 'Subject','ChangePoint']), train_data['Activity'], time_steps, step)
    X_val, y_val = create_dataset(val_data.drop(columns=['Activity', 'Subject', 'ChangePoint']), val_data['Activity'], time_steps, step)
        
    X_train_std, X_val_std, scaler = standardize_data_fine_tuning(X_train, X_val)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_std, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_std, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=False, drop_last=False)
    
    unique_labels = np.unique(y_train)
    num_classes = len(unique_labels)
    num_signals = X_train.shape[1]
    out_channels1 = config['out_channels1']
    out_channels2 = config['out_channels2']
    out_channels3 = config['out_channels3']
    out_channels4 = config['out_channels4']
    lstm_out_channels = config['lstm_out_channels']
    epochs = config['epoch']
    fen_lr = config['fen_lr']
    fln_lr = config['fln_lr']
    
    # initialize FEN/FLN
    fen_model = FEN(1, out_channels1, out_channels2, out_channels3, out_channels4).to(device)
    fln_model = FLN(out_channels4*num_signals, lstm_out_channels, num_classes).to(device)
    
    weights_folder = 'weights_after_tf_8_labels'
    fen_weights_file = os.path.join(weights_folder, f'best_fen_weights_2_0.pth')
    print(fen_weights_file)
    if os.path.exists(fen_weights_file):
        fen_model.load_state_dict(torch.load(fen_weights_file))
        print("FEN weights loaded")
    for param in fen_model.parameters():
        param.requires_grad = True
    
    fen_model, fln_model, best_fen_model_state, best_fln_model_state, best_acc, best_loss = training_loop(train_loader, val_loader, fen_model, fln_model, epochs, device, fen_lr, fln_lr, config['fen_wd'], config['fln_wd'], 10)

    if best_fen_model_state is not None:
        fen_model.load_state_dict(best_fen_model_state)
        fln_model.load_state_dict(best_fln_model_state)
        
        # Load and prepare test data
        segment_indices = load_segment_indices_for_subject(test_subject)
        segments = [(0, segment_indices[0] - 1)] if segment_indices[0] > 0 else []
        segments += [(segment_indices[i], segment_indices[i+1] - 1) for i in range(len(segment_indices) - 1)]
        segments += [(segment_indices[-1], len(test_data) - 1)] if segment_indices[-1] < len(test_data) else []

        if segments[0][1] - segments[0][0] + 1 < 400:
            segments[1] = (segments[0][0], segments[1][1])
            segments.pop(0)
        last_index = len(segments) - 1
        if segments[last_index][1] - segments[last_index][0] + 1 < 400:
            segments[last_index - 1] = (segments[last_index - 1][0], segments[last_index][1])
            segments.pop()

        test_dataset = SegmentDataset(test_data, segments, scaler)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
         
        test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(test_loader, fen_model, fln_model, device, time_steps, step)
        
        results_summary[test_subject] = {
        "Best Validation Loss": best_loss,
        "Best Validation Accuracy": best_acc,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "F1-Score": test_f1,
        "Precision": test_precision,
        "Recall": test_recall
        }
        print(f'Test Loss for subject {test_subject}: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%, '
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

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
        
def main():
    root_dir = './'
    data_file = 'Pre_processed_OutSense.pkl'
    full_path = os.path.join(root_dir, data_file)
    data = load_data(full_path)
    
    all_subjects = np.unique(data['Subject'])
    
    parameter_grid = {
            'out_channels1': [64],
            'out_channels2': [128],
            'out_channels3': [64],
            'out_channels4': [256],
            'lstm_out_channels': [256],
            'fen_lr': [0.001],
            'fln_lr': [0.0001],
            'fen_wd': [7e-5],
            'fln_wd': [3e-6],  
            'epoch': [100],
    }
    all_results = {}
    keys, values = zip(*parameter_grid.items())
    f1_scores = [] 


    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        config_f1_scores = []

        print(f"\nRunning with configuration: {config}")
        
        for test_subject in all_subjects:
            print(f"\nRunning LOSO for test subject: {test_subject}")
            sys.stdout.flush()
            final_results = run_experiment(config, data, test_subject)
            all_results[f"{config}_{test_subject}"] = final_results

            config_f1_scores.append(final_results[test_subject]['F1-Score'])

        # Store F1 scores for later analysis
        f1_scores.append(config_f1_scores)

        # Calculate and print the average and standard deviation of F1 scores for this configuration
        mean_f1 = np.mean(config_f1_scores)
        std_f1 = np.std(config_f1_scores)
        print(f"Configuration {config}: F1 Scores: {config_f1_scores}")
        print(f"Average F1 Score: {mean_f1:.4f}, Standard Deviation: {std_f1:.4f}\n")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time() 
    total_time_seconds = end_time - start_time
    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60) 
    print(f"\nTotal Execution Time: {hours} hours and {minutes} minutes")