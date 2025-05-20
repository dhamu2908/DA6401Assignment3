import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import random
import wandb
import argparse

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Special tokens
START_SYMBOL = '<'
END_SYMBOL = '>'
PADDING_SYMBOL = '_'
TEACHER_FORCING_PROB = 0.5

class SequenceProcessor:
    def __init__(self):
        self.char_to_idx = {START_SYMBOL: 0, END_SYMBOL: 1, PADDING_SYMBOL: 2}
        self.idx_to_char = {0: START_SYMBOL, 1: END_SYMBOL, 2: PADDING_SYMBOL}
        self.vocab_size = 3

    def build_vocabulary(self, sequences):
        for seq in sequences:
            for char in seq:
                if char not in self.char_to_idx:
                    idx = self.vocab_size
                    self.char_to_idx[char] = idx
                    self.idx_to_char[idx] = char
                    self.vocab_size += 1

    def pad_sequence(self, seq, max_len):
        padded = START_SYMBOL + seq + END_SYMBOL
        padded = padded[:max_len] + PADDING_SYMBOL * (max_len - len(padded))
        return padded

    def sequence_to_tensor(self, seq):
        return torch.tensor([self.char_to_idx[char] for char in seq], device=DEVICE)

class SequenceDataset(Dataset):
    def __init__(self, source_sequences, target_sequences, processor, max_src_len, max_tgt_len):
        self.source = [processor.pad_sequence(seq, max_src_len) for seq in source_sequences]
        self.target = [processor.pad_sequence(seq, max_tgt_len) for seq in target_sequences]
        self.processor = processor
        
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        src_tensor = self.processor.sequence_to_tensor(self.source[idx])
        tgt_tensor = self.processor.sequence_to_tensor(self.target[idx])
        return src_tensor, tgt_tensor

class NeuralSequenceModel:
    @staticmethod
    def create_rnn_cell(cell_type, input_size, hidden_size, num_layers):
        cell_types = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU
        }
        return cell_types[cell_type](input_size, hidden_size, num_layers, batch_first=True)

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_state, encoder_outputs):
        # Calculate attention scores
        scores = self.score_proj(torch.tanh(
            self.query_proj(decoder_state.unsqueeze(1)) + 
            self.key_proj(encoder_outputs)
        ))
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.transpose(1, 2), encoder_outputs)
        return context, weights

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, cell_type):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = NeuralSequenceModel.create_rnn_cell(
            cell_type, embed_dim, hidden_dim, num_layers)
        
    def forward(self, x, hidden_state):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded, hidden_state)
        return outputs, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=DEVICE)

class SequenceDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, cell_type):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = AttentionMechanism(hidden_dim)
        self.rnn = NeuralSequenceModel.create_rnn_cell(
            cell_type, embed_dim + hidden_dim, hidden_dim, num_layers)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, prev_hidden, encoder_outputs):
        embedded = self.embedding(x)
        context, attn_weights = self.attention(prev_hidden[-1], encoder_outputs)
        rnn_input = torch.cat([embedded, context], dim=2)
        outputs, hidden = self.rnn(rnn_input, prev_hidden)
        logits = F.log_softmax(self.output_layer(outputs), dim=2)
        return logits, hidden, attn_weights

class SequenceTrainer:
    def __init__(self, encoder, decoder, optimizer, loss_fn):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
    def train_step(self, src_batch, tgt_batch, teacher_forcing):
        batch_size = src_batch.size(0)
        
        # Initialize hidden state
        hidden = self.encoder.init_hidden(batch_size)
        if isinstance(hidden, tuple):  # For LSTM
            hidden = (hidden, self.encoder.init_hidden(batch_size))
            
        # Encode input sequence
        encoder_outputs, hidden = self.encoder(src_batch.unsqueeze(1), hidden)
        
        # Prepare decoder inputs
        decoder_input = torch.full((batch_size, 1), 
                                  self.decoder.embedding.weight.shape[0]-3, 
                                  device=DEVICE)
        loss = 0
        correct = 0
        
        # Decode sequence
        for t in range(tgt_batch.size(1)):
            logits, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            _, topi = logits.topk(1)
            decoder_input = topi.squeeze().detach()
            
            # Calculate loss
            loss += self.loss_fn(logits.squeeze(1), tgt_batch[:, t])
            correct += (topi.squeeze() == tgt_batch[:, t]).sum().item()
            
            # Teacher forcing
            if teacher_forcing and random.random() < TEACHER_FORCING_PROB:
                decoder_input = tgt_batch[:, t].unsqueeze(1)
                
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), correct

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            # Evaluation logic here
            pass
    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)

def prepare_data_loaders(train_src, train_tgt, val_src, val_tgt, batch_size):
    processor = SequenceProcessor()
    max_src_len = max(len(s) for s in train_src) + 2
    max_tgt_len = max(len(t) for t in train_tgt) + 2
    
    processor.build_vocabulary(train_src + train_tgt)
    
    train_dataset = SequenceDataset(train_src, train_tgt, processor, max_src_len, max_tgt_len)
    val_dataset = SequenceDataset(val_src, val_tgt, processor, max_src_len, max_tgt_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, processor

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--cell_type", choices=["RNN", "LSTM", "GRU"], default="LSTM")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(args.train_path, header=None)
    val_df = pd.read_csv(args.val_path, header=None)
    
    train_src, train_tgt = train_df[0].tolist(), train_df[1].tolist()
    val_src, val_tgt = val_df[0].tolist(), val_df[1].tolist()

    # Prepare data loaders
    train_loader, val_loader, processor = prepare_data_loaders(
        train_src, train_tgt, val_src, val_tgt, args.batch_size)

    # Initialize models
    encoder = SequenceEncoder(processor.vocab_size, args.embed_dim, 
                            args.hidden_dim, args.num_layers, args.cell_type).to(DEVICE)
    decoder = SequenceDecoder(processor.vocab_size, args.embed_dim, 
                            args.hidden_dim, args.num_layers, args.cell_type).to(DEVICE)

    # Initialize optimizer and loss
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                         lr=args.learning_rate)
    loss_fn = nn.NLLLoss()

    # Train model
    trainer = SequenceTrainer(encoder, decoder, optimizer, loss_fn)
    for epoch in range(args.epochs):
        # Training and evaluation logic here
        pass

if __name__ == "__main__":
    main()