import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import argparse
import wandb

class SequenceProcessor:
    """Handles vocabulary creation and sequence processing"""
    def __init__(self):
        self.special_tokens = {
            'start': '<', 
            'end': '>',
            'pad': '_'
        }
        self.build_base_vocab()
        
    def build_base_vocab(self):
        """Initialize with special tokens"""
        self.char2idx = {v: i for i, v in enumerate(self.special_tokens.values())}
        self.idx2char = {i: v for v, i in self.char2idx.items()}
        
    def build_vocab(self, sequences):
        """Add all unique characters from sequences to vocabulary"""
        for seq in sequences:
            for char in seq:
                if char not in self.char2idx:
                    idx = len(self.char2idx)
                    self.char2idx[char] = idx
                    self.idx2char[idx] = char
    
    def process_sequences(self, sequences, max_len):
        """Convert raw sequences to padded tensors"""
        processed = []
        for seq in sequences:
            # Add start and end tokens
            seq = self.special_tokens['start'] + seq + self.special_tokens['end']
            # Truncate or pad
            seq = seq[:max_len] + self.special_tokens['pad'] * (max_len - len(seq))
            # Convert to indices
            indices = [self.char2idx[c] for c in seq]
            processed.append(torch.tensor(indices))
        return torch.stack(processed)

class Seq2SeqDataset(Dataset):
    """Custom dataset for sequence pairs"""
    def __init__(self, src_sequences, tgt_sequences, processor, max_src_len, max_tgt_len):
        self.src = processor.process_sequences(src_sequences, max_src_len)
        self.tgt = processor.process_sequences(tgt_sequences, max_tgt_len)
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

class Encoder(nn.Module):
    """Encoder module with configurable RNN type"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        rnn_cells = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        self.rnn = rnn_cells[cell_type](
            embed_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, x, hidden):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        weight = next(self.parameters())
        if isinstance(self.rnn, nn.LSTM):
            return (weight.new_zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size),
                    weight.new_zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size))
        return weight.new_zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size)

class Attention(nn.Module):
    """Attention mechanism for decoder"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # Expand hidden state to match encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # Return attention weights and context vector
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    """Decoder module with attention"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)
        
        rnn_cells = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        self.rnn = rnn_cells[cell_type](
            embed_dim + hidden_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x, hidden, encoder_outputs):
        # Process input
        embedded = self.dropout(self.embedding(x))
        
        # Calculate attention weights and context vector
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        # Combine embedded input and context
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # RNN step
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Final prediction
        output = torch.cat((output, context), dim=2)
        prediction = self.fc(output)
        
        return prediction, hidden, attn_weights

class Seq2SeqTrainer:
    """Handles model training and evaluation"""
    def __init__(self, encoder, decoder, optimizer, device):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=2)  # Ignore padding index
        
    def train_step(self, src, tgt, teacher_forcing_ratio=0.5):
        self.optimizer.zero_grad()
        
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features
        
        # Initialize hidden state
        hidden = self.encoder.init_hidden(batch_size)
        
        # Encode input
        encoder_outputs, hidden = self.encoder(src.unsqueeze(1), hidden)
        
        # Prepare decoder input
        decoder_input = tgt[:, 0].unsqueeze(1)  # Start token
        loss = 0
        correct = 0
        
        # Decode sequence
        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            
            # Calculate loss
            loss += self.criterion(output.squeeze(1), tgt[:, t])
            
            # Get predictions
            _, topi = output.topk(1)
            correct += (topi.squeeze() == tgt[:, t]).sum().item()
            
            # Teacher forcing
            decoder_input = tgt[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else topi.detach()
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        
        return loss.item() / tgt_len, correct / (batch_size * (tgt_len - 1))
    
    def evaluate(self, dataloader):
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        total_correct = 0
        total = 0
        
        with torch.no_grad():
            for src, tgt in dataloader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                batch_size = src.size(0)
                tgt_len = tgt.size(1)
                
                # Initialize hidden state
                hidden = self.encoder.init_hidden(batch_size)
                
                # Encode input
                encoder_outputs, hidden = self.encoder(src.unsqueeze(1), hidden)
                
                # Prepare decoder input
                decoder_input = tgt[:, 0].unsqueeze(1)
                batch_loss = 0
                batch_correct = 0
                
                # Decode sequence
                for t in range(1, tgt_len):
                    output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
                    
                    # Calculate loss
                    batch_loss += self.criterion(output.squeeze(1), tgt[:, t])
                    
                    # Get predictions
                    _, topi = output.topk(1)
                    batch_correct += (topi.squeeze() == tgt[:, t]).sum().item()
                    
                    # Next input
                    decoder_input = topi.detach()
                
                total_loss += batch_loss.item()
                total_correct += batch_correct
                total += batch_size * (tgt_len - 1)
        
        return total_loss / len(dataloader), total_correct / total

def load_data(path):
    """Load data from CSV file"""
    df = pd.read_csv(path, header=None)
    return df[0].tolist(), df[1].tolist()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Sequence-to-sequence model training")
    parser.add_argument("--train_path", required=True, help="Path to training data")
    parser.add_argument("--val_path", required=True, help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--cell_type", choices=["RNN", "LSTM", "GRU"], default="LSTM")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--teacher_forcing", type=float, default=0.5)
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="seq2seq", config=vars(args))
    
    # Load and process data
    train_src, train_tgt = load_data(args.train_path)
    val_src, val_tgt = load_data(args.val_path)
    
    processor = SequenceProcessor()
    processor.build_vocab(train_src + train_tgt)
    
    # Calculate max lengths
    max_src_len = max(len(s) for s in train_src) + 2  # +2 for start/end tokens
    max_tgt_len = max(len(t) for t in train_tgt) + 2
    
    # Create datasets
    train_dataset = Seq2SeqDataset(train_src, train_tgt, processor, max_src_len, max_tgt_len)
    val_dataset = Seq2SeqDataset(val_src, val_tgt, processor, max_src_len, max_tgt_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(len(processor.char2idx), args.embed_dim, args.hidden_dim, 
                     args.num_layers, args.cell_type, args.dropout).to(device)
    decoder = Decoder(len(processor.char2idx), args.embed_dim, args.hidden_dim,
                     args.num_layers, args.cell_type, args.dropout).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(encoder, decoder, optimizer, device)
    
    # Training loop
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        
        epoch_loss = 0
        epoch_acc = 0
        
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            loss, acc = trainer.train_step(src, tgt, args.teacher_forcing)
            epoch_loss += loss
            epoch_acc += acc
        
        # Calculate validation metrics
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
            "train_acc": epoch_acc / len(train_loader),
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f} | Acc: {epoch_acc/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()