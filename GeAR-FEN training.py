import os
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from scipy import stats
import pickle
from model import FEN, FLN
import time
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.metrics import f1_score, precision_score, recall_score
import optuna

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
def training_loop(dataset, train_loader, val_loader, fen_model, fln_model, epochs, device, fln_weights_dict, fen_lr, fln_lr, patience):
    fen_optimizer = optim.Adam(fen_model.parameters(), lr=fen_lr, weight_decay=1e-5)
    fln_optimizer = optim.Adam(fln_model.parameters(), lr=fln_lr, weight_decay=1e-5)
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
            fln_weights_dict[dataset] = copy.deepcopy(fln_model.state_dict())
            fen_state = copy.deepcopy(fen_model.state_dict())
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Best Validation Loss: {best_loss:.4f} with Accuracy: {best_acc:.4f}%")
    return fen_model, fln_model, fln_weights_dict, fen_state, best_acc, best_loss

# Evaluate the trained models on test data
def evaluate_model(test_loader, fen_model, fln_model, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_labels = []
    all_predictions = []
    
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

    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * total_correct / total_samples
    
    # Calculate weighted metrics
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted')
    
    return test_loss, test_accuracy, f1, precision, recall

def save_weights(fen_state, fln_weights_dict, dataset_name, weights_folder,outchannels1,outchannels2,outchannels3,outchannels4,lstm_out_channels):
    fen_weights_path = os.path.join(weights_folder, f'CNNBiLSTMAttn_fen_weights_{outchannels1}_{outchannels2}_{outchannels3}_{outchannels4}_{lstm_out_channels}.pth')
    fln_weights_path = os.path.join(weights_folder, f'CNNBiLSTMAttn_fln_weights_{dataset_name}_{outchannels1}_{outchannels2}_{outchannels3}_{outchannels4}_{lstm_out_channels}.pth')
    torch.save(fen_state, fen_weights_path)
    torch.save(fln_weights_dict[dataset_name], fln_weights_path)

# Standardization of time-series data before training
def standardize_data(X_train, X_val, X_test1, X_test2):
    # Flatten the 3D arrays to 2D for standardization
    nsamples, nfeatures, ntimesteps = X_train.shape
    X_train_flat = X_train.reshape((nsamples * ntimesteps, nfeatures))
    X_val_flat = X_val.reshape((X_val.shape[0] * X_val.shape[2], X_val.shape[1]))
    X_test1_flat = X_test1.reshape((X_test1.shape[0] * X_test1.shape[2], X_test1.shape[1]))
    X_test2_flat = X_test2.reshape((X_test2.shape[0] * X_test2.shape[2], X_test2.shape[1]))

    # Standardize the flattened data
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test1_flat = scaler.transform(X_test1_flat)
    X_test2_flat = scaler.transform(X_test2_flat)

    # Reshape back to the original 3D shape
    X_train_std = X_train_flat.reshape((nsamples, nfeatures, ntimesteps))
    X_val_std = X_val_flat.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    X_test1_std = X_test1_flat.reshape((X_test1.shape[0], X_test1.shape[1], X_test1.shape[2]))
    X_test2_std = X_test2_flat.reshape((X_test2.shape[0], X_test2.shape[1], X_test2.shape[2]))

    return X_train_std, X_val_std, X_test1_std, X_test2_std

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

# Fine_tuning_after_early_stopping_iterations
def fine_tuning(seed_num, config, data_splits,split_index):
    print('Start Fine Tuning')
    print(f"Running experiment with config: {config}")
    fln_weights_dict = {}
    fen_state = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    root_dir = '../Datapool_new/'
    print([f for f in os.listdir(root_dir) if f.endswith('.pkl')])
    all_files = [f for f in os.listdir(root_dir) if f.endswith('.pkl')]
    fine_tuning_results_summary = {}
        
    for chosen_file in all_files:
        print(f"Running experiment with seed: {seed_num} for {chosen_file}")
        setup_seed(seed_num)
        with open(os.path.join(root_dir, chosen_file), 'rb') as f:
            data = pickle.load(f)
        dataset_name = os.path.splitext(chosen_file)[0]

        # data split
        subjects = data['subject'].unique()

        print(f"dataset:", dataset_name)
        
        train_subjects, val_subjects, test1_subjects, test2_subjects = data_splits[dataset_name][split_index]
        combined_train_subjects = np.concatenate((train_subjects, test1_subjects))
        
        print(combined_train_subjects,val_subjects,test2_subjects)

        train_data = data[data['subject'].isin(combined_train_subjects)]
        val_data = data[data['subject'].isin(val_subjects)]
        test_data = data[data['subject'].isin(test2_subjects)]

        time_steps = 400
        step = 200

        X_train, y_train = create_dataset(train_data.drop(columns=['label', 'subject']), train_data['label'], time_steps, step)
        X_val, y_val = create_dataset(val_data.drop(columns=['label', 'subject']), val_data['label'], time_steps, step)
        X_test, y_test = create_dataset(test_data.drop(columns=['label', 'subject']), test_data['label'], time_steps, step)

        X_train_std, X_val_std, X_test_std = standardize_data_fine_tuning(X_train, X_val, X_test)
        # Use the standardized data for creating data loaders
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_std, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=True, drop_last=False)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val_std, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=False, drop_last=False)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test_std, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=False, drop_last=False)

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
        
        weights_folder = f'saved_weights_0527_{split_index}'
        fen_weights_file = os.path.join(weights_folder, f'CNNBiLSTMAttn_fen_weights_{out_channels1}_{out_channels2}_{out_channels3}_{out_channels4}_{lstm_out_channels}.pth')
        print(fen_weights_file)
        if os.path.exists(fen_weights_file):
            fen_model.load_state_dict(torch.load(fen_weights_file))
            print("FEN weights loaded")

        # Attempt to load FLN weights if available
        fln_weights_file = os.path.join(weights_folder, f'CNNBiLSTMAttn_fln_weights_{dataset_name}_{out_channels1}_{out_channels2}_{out_channels3}_{out_channels4}_{lstm_out_channels}.pth')
        if os.path.exists(fln_weights_file):
            fln_model.load_state_dict(torch.load(fln_weights_file))
            print(f"FLN weights loaded for {dataset_name}")

        fen_model, fln_model, fln_weights_dict, fen_state, best_acc, best_loss = training_loop(dataset_name, train_loader, val_loader, fen_model, fln_model, epochs, device, fln_weights_dict, fen_lr, fln_lr, 10)
        
        if fen_state is not None:
            fen_model.load_state_dict(fen_state)
            fln_model.load_state_dict(fln_weights_dict[dataset_name])

            test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(test_loader, fen_model, fln_model, device)
            
            fine_tuning_results_summary[dataset_name] = {
            "Best Validation Loss": best_loss,
            "Best Validation Accuracy": best_acc,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "F1-Score": test_f1,
            "Precision": test_precision,
            "Recall": test_recall
            }
                    
            print(f'Fine Tuning: Test Loss for {dataset_name}: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, '
                f'F1-Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')          
    return fine_tuning_results_summary

# Running the entire experimental setup                
def run_experiment(config, data_splits,split_ind):
    print(f"split_index: {split_ind}, Running experiment with config: {config}")
    fln_weights_dict = {}
    fen_state = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    setup_seed(1)
    root_dir = '../Datapool_new/'
    all_files = [f for f in os.listdir(root_dir) if f.endswith('.pkl')]
    print(all_files)
    results_summary = {}
    
    iteration_scores = []
    dataset_f1_scores = {}
    highest_f1_score = 0
    decline_count = 0
    
    test2_loaders = {}
    num_signals_dict = {}
    num_classes_dict = {}
    
    best_fen_state = None
    best_fln_weights_dict = {}
    for iteration in range(20):
        random.shuffle(all_files)
        print(f"Starting iteration {iteration + 1} with files: {all_files}")
        current_iteration_f1_scores = []
        iteration_dataset_results = []
        
        for chosen_file in all_files:
            with open(os.path.join(root_dir, chosen_file), 'rb') as f:
                data = pickle.load(f)
            dataset_name = os.path.splitext(chosen_file)[0]

            if dataset_name not in dataset_f1_scores:
                dataset_f1_scores[dataset_name] = []
            
            print(f"dataset:", dataset_name)
            
            split_index = split_ind
            train_subjects, val_subjects, test1_subjects, test2_subjects = data_splits[dataset_name][split_index]
            print(train_subjects, val_subjects, test1_subjects, test2_subjects)

            train_data = data[data['subject'].isin(train_subjects)]
            val_data = data[data['subject'].isin(val_subjects)]
            test1_data = data[data['subject'].isin(test1_subjects)]
            test2_data = data[data['subject'].isin(test2_subjects)]

            time_steps = 400
            step = 200

            X_train, y_train = create_dataset(train_data.drop(columns=['label', 'subject']), train_data['label'], time_steps, step)
            X_val, y_val = create_dataset(val_data.drop(columns=['label', 'subject']), val_data['label'], time_steps, step)
            X_test1, y_test1 = create_dataset(test1_data.drop(columns=['label', 'subject']), test1_data['label'], time_steps, step)
            X_test2, y_test2 = create_dataset(test2_data.drop(columns=['label', 'subject']), test2_data['label'], time_steps, step)

            X_train_std, X_val_std, X_test1_std, X_test2_std = standardize_data(X_train, X_val, X_test1, X_test2)
            # Use the standardized data for creating data loaders
            train_loader = DataLoader(TensorDataset(torch.tensor(X_train_std, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=True, drop_last=False)
            val_loader = DataLoader(TensorDataset(torch.tensor(X_val_std, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=False, drop_last=False)
            test1_loader = DataLoader(TensorDataset(torch.tensor(X_test1_std, dtype=torch.float32), torch.tensor(y_test1, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=False, drop_last=False)
            test2_loader = DataLoader(TensorDataset(torch.tensor(X_test2_std, dtype=torch.float32), torch.tensor(y_test2, dtype=torch.long).squeeze(1)), batch_size=64, shuffle=False, drop_last=False)

            if iteration == 0:
                test2_loaders[dataset_name] = test2_loader

            unique_labels = np.unique(y_train)
            num_classes = len(unique_labels)
            num_signals = X_train.shape[1]
            
            num_signals_dict[dataset_name] = num_signals
            num_classes_dict[dataset_name] = num_classes
            
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
            if fen_state is not None:
                fen_model.load_state_dict(fen_state)
                
            fln_model = FLN(out_channels4*num_signals, lstm_out_channels, num_classes).to(device)
            if dataset_name in fln_weights_dict:
                fln_model.load_state_dict(fln_weights_dict[dataset_name])  
                
            fen_model, fln_model, fln_weights_dict, fen_state, best_acc, best_loss = training_loop(dataset_name, train_loader, val_loader, fen_model, fln_model, epochs, device, fln_weights_dict, fen_lr, fln_lr, 10)
        
            fen_model.load_state_dict(fen_state)
            fln_model.load_state_dict(fln_weights_dict[dataset_name])
            
            _, _, test1_f1, _, _ = evaluate_model(test1_loader, fen_model, fln_model, device)
            current_iteration_f1_scores.append(test1_f1)
            iteration_dataset_results.append((dataset_name, test1_f1))
            dataset_f1_scores[dataset_name].append(test1_f1)     
              
        # Calculate the average micro-average F1 score across all datasets for this iteration
        average_f1_score = np.mean(current_iteration_f1_scores)
        iteration_scores.append(average_f1_score)
        print(f"Average F1 Score for iteration {iteration + 1}: {average_f1_score:.4f}")
        
        for result in iteration_dataset_results:
            print(f"F1 Score for {result[0]}: {result[1]:.4f}")
            
        # Update highest F1 score and decline count
        if average_f1_score > highest_f1_score:
            highest_f1_score = average_f1_score
            decline_count = 0
            best_fen_state = copy.deepcopy(fen_state)
            best_fln_weights_dict = copy.deepcopy(fln_weights_dict)
        else:
            decline_count += 1

        if decline_count >= 4:
            print("Early stopping triggered due to five declines in performance below the highest recorded F1 score.")
            break
    
    print("Average F1 Scores for all iterations:")
    for i, score in enumerate(iteration_scores):
        print(f"Iteration {i + 1}: {score:.4f}")

    # Print each dataset's F1 scores across all iterations
    print("All dataset F1 scores for each iteration:")
    for dataset_name, scores in dataset_f1_scores.items():
        scores_formatted = ', '.join(f"{score:.4f}" for score in scores)
        print(f"{dataset_name}: [{scores_formatted}]")    
        
    # Final Evaluation using saved test2_loaders and model weights
    weights_folder = f'saved_weights_0527_{split_index}'
    os.makedirs(weights_folder, exist_ok=True)    
    for dataset_name, test2_loader in test2_loaders.items():
        fen_model = FEN(1, config['out_channels1'], config['out_channels2'],config['out_channels3'],config['out_channels4']).to(device)
        fln_model = FLN(config['out_channels4']*num_signals_dict[dataset_name],config['lstm_out_channels'], num_classes_dict[dataset_name]).to(device)
        fen_model.load_state_dict(best_fen_state)
        fln_model.load_state_dict(best_fln_weights_dict[dataset_name])

        save_weights(best_fen_state, best_fln_weights_dict, dataset_name,weights_folder,config['out_channels1'], config['out_channels2'],config['out_channels3'],config['out_channels4'],config['lstm_out_channels'])

        test2_loss, test2_accuracy, test2_f1, test2_precision, test2_recall = evaluate_model(test2_loader, fen_model, fln_model, device)

        results_summary[dataset_name] = {
            "Test2 Loss": test2_loss,
            "Test2 Accuracy": test2_accuracy,
            "Test2 F1-Score": test2_f1,
            "Test2 Precision": test2_precision,
            "Test2 Recall": test2_recall
        }
        print(f'Final Test Results for {dataset_name}: Loss: {test2_loss:.4f}, Accuracy: {test2_accuracy:.2f}%, F1-Score: {test2_f1:.4f}')
        
    fine_tuning_results_summary = fine_tuning(1,config,data_splits,split_ind)
    print("\n")
    for dataset, metrics in fine_tuning_results_summary.items():
        print(f"{dataset}: Best Val Loss: {metrics['Best Validation Loss']:.4f}, Best Val Accuracy: {metrics['Best Validation Accuracy']:.2f}%, Test Loss: {metrics['Test Loss']:.4f}, Test Accuracy: {metrics['Test Accuracy']:.2f}%, F1-Score: {metrics['F1-Score']:.4f}")   
    return results_summary, fine_tuning_results_summary

def pretty_print_results(results, split_index):
    print(f"\nResults for split index: {split_index}")
    for config, datasets_results in results.items():
        print(f"Configuration: {config}")
        for dataset_name, metrics in datasets_results.items():
            print(f"  Dataset: {dataset_name}")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        print("\n")
        
# Set grid search
def objective(trial, split_index, data_splits):
    parameter_grid = {
        'out_channels1': trial.suggest_categorical('out_channels1', [16,32,64,128]),
        'out_channels2': trial.suggest_categorical('out_channels2', [32,64,128,256]),
        'out_channels3': trial.suggest_categorical('out_channels3', [32,64,128,256]),
        'out_channels4': trial.suggest_categorical('out_channels4', [32,64,128,256]),
        'lstm_out_channels': trial.suggest_categorical('lstm_out_channels', [32,64,128,256]),
        'fen_lr': 0.001,
        'fln_lr': 0.001,
        'epoch': 100,
    }
    
    results_summary,fine_tuning_results_summary = run_experiment(parameter_grid, data_splits, split_index)
    f1_scores = [metrics['F1-Score'] for metrics in fine_tuning_results_summary.values()]
    
    return np.mean(f1_scores)

def optimize_for_split_index(split_index, top_n=5):
    print(f"Optimizing for split_index: {split_index}")
    
    with open('data_splits.pkl', 'rb') as f:
        data_splits = pickle.load(f)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, split_index, data_splits), n_trials=100)
    
    print(f"Best hyperparameters for split_index {split_index}: ", study.best_params)
    
    trials_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    top_trials = trials_df.nlargest(top_n, 'value')
    
    top_params = []
    for index, row in top_trials.iterrows():
        params = {key: row[key] for key in row.index if key.startswith('params_')}
        params['Average F1 Score'] = row['value']
        top_params.append(params)
    
    return top_params

overall_start_time = time.time()
best_params_per_split = {}

for split_index in range(1,2):
    best_params = optimize_for_split_index(split_index, top_n=5)
    best_params_per_split[split_index] = best_params

print("\nTop 5 hyperparameters for each split_index:")
for split_index, params in best_params_per_split.items():
    print(f"split_index {split_index}:")
    for i, param in enumerate(params, 1):
        print(f"  Rank {i}: {param}")

# Calculate total execution time
overall_end_time = time.time()
total_time_seconds = overall_end_time - overall_start_time
hours, minutes = divmod(total_time_seconds, 3600)
minutes, seconds = divmod(minutes, 60)
# pretty_print_results(all_results)
print(f"\nTotal Execution Time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")