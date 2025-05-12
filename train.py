from sklearn.metrics import ndcg_score
import evaluate
import argparse
from data_preparation import Bert4RecDataset
from torch.utils.data import DataLoader
from model import BERT4Rec
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

def train(model, optimizer, loss_fn, train_dataloader, device):
    print ('training')
    model.train()
    train_loss = 0
    for step, batch in enumerate(train_dataloader):
        tokens = batch['tokens'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = model(tokens) # [B, T, vocab_size]
        # B, T, d_model = logits.shape
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        train_loss += loss

        loss.backward()
        optimizer.step()

    return train_loss/len(train_dataloader)

@torch.no_grad()
def eval(model, loss_fn, eval_dataloader, device):
    model.eval()
    eval_loss = 0
    metric_accuracy = evaluate.load("accuracy")
    ndcg_scores = []
    for step, batch in enumerate(eval_dataloader):
        tokens = batch['tokens'].to(device)
        labels = batch['labels'].to(device)
        neg_sample = batch['neg_sample']

        B, N = neg_sample.shape

        logits = model(tokens) # [B, T, vocab_size]
        # B, T, V = logits.shape
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        eval_loss += loss

        targets = labels[:,-1].unsqueeze(-1)
        # print (targets.shape) # B, K
        B, K = targets.shape
        candidates = torch.cat((targets, neg_sample), dim=1)
        # print (candidates.shape) # B, K+N

        logits = logits[:, -1, :]  # last position
        metric_accuracy.add_batch(predictions=logits.argmax(dim=1), references=targets)
        _ndcg10 = ndcg_score(np.broadcast_to(np.array([1]*K+[0]*N),(B,K+N)), logits.gather(1, candidates).cpu())
        ndcg_scores.append(_ndcg10)
    accuracy = metric_accuracy.compute()
    return eval_loss/len(eval_dataloader), np.mean(ndcg_scores), accuracy


def build_args():
    parser = argparse.ArgumentParser(description="Bert4Rec Args")
    parser.add_argument("--data_dir", type=str, default="./data/ml-1m")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_len", type=int, default=32)
    parser.add_argument("--mask_prob", type=int, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()
    return args

def bert4rec_training(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # Load Data
    train_data = Bert4RecDataset(args.data_dir, args.context_len, args.mask_prob, split='train')
    vocab_size = train_data.get_vocab_size()
    eval_data = Bert4RecDataset(args.data_dir, args.context_len, args.mask_prob, split='eval')

    # BERT4Rec Model
    bert4rec_model = BERT4Rec(
        vocab_size, 
        context_len=args.context_len, 
        d_model=args.hidden_dim, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers, 
        dropout=args.dropout
        )
    bert4rec_model.to(device)

    train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_data, batch_size=args.batch_size, shuffle=True)
    # print (len(train_dataloader), len(eval_dataloader))

    optimizer = AdamW(bert4rec_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(args.num_epochs):
        train_loss = train(bert4rec_model, optimizer, loss_fn, train_dataloader, device)
        print (f'train loss at epoch {epoch} is {train_loss}')
        eval_loss, eval_ndcg10, accuracy = eval(bert4rec_model, loss_fn, eval_dataloader, device)
        print (f'eval loss at epoch {epoch} is {eval_loss}')
        print (f'eval NDCG@10 at epoch {epoch} is {eval_ndcg10}')



if __name__ == "__main__":
    args = build_args()
    print ('running Bert4Rec with following args')
    print (args)

    bert4rec_training(args)