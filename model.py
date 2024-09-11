import torch
import torch.nn as nn
import torch.nn.functional as F

class FEN(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4):
        super(FEN, self).__init__()
        kernel_size = 3
        padding = 1
        maxpool_kernel_size = 2
        dropout_rate = 0.2
        
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels1, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels1, out_channels2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels2, out_channels3, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels3, out_channels4, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=2),
            nn.Dropout(dropout_rate),
        )
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)
        return x

class ResBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ResBLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)  # For bidirectional output
        self.transform = nn.Linear(input_size, hidden_size * 2)  # Transform input size to match LSTM output

    def forward(self, x):
        # Transform input to match the output size of LSTM
        residual = self.transform(x)
        output, (hn, cn) = self.lstm(x)
        output = self.layer_norm(output)
        return output + residual
    
class AttentionLayer(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights_layer = nn.Linear(lstm_hidden_size, 1)
    
    def forward(self, x):
        # x shape is (batch, seq_len, num_directions * hidden_size)
        attention_scores = self.attention_weights_layer(x).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_feature_vector = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        return weighted_feature_vector, attention_weights

class FLN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FLN, self).__init__()
        self.res_bilstm = ResBLSTM(input_size, hidden_size, num_layers=2)
        self.attention_layer = AttentionLayer(hidden_size * 2)  # Adjusted for bidirectional output
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Adjusted for bidirectional output
    
    def forward(self, x):
        x = self.res_bilstm(x)
        attention_output, attention_weights = self.attention_layer(x)
        classification_output = self.fc(attention_output)
        return classification_output