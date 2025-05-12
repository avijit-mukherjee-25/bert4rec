import pandas as pd
import numpy as np
import numpy as np
import torch
from torch.utils.data import Dataset

class Bert4RecDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            context_len, 
            mask_prob,
            split
            ):
        super(Bert4RecDataset, self).__init__()
        self.context_len = context_len
        self.mask_prob = mask_prob
        self.data, self.popular_movies, self.vocab_size = self.prep_data(data_dir)
        self.split = split

    def prep_data(self, data_dir):
        ratings = pd.read_csv(
            f'{data_dir}/ratings.dat',sep='::',
            engine='python',
            encoding='latin-1', 
            header=None,
            names=['UserID', 'MovieID', 'Rating', 'Timestamp']
        )
        movie_idx = {}
        for idx, movie in enumerate(ratings.MovieID.unique()):
            movie_idx[movie] = idx+1
        ratings['movie_idx'] = ratings.MovieID.apply(lambda x: movie_idx[x])

        data = ratings.groupby('UserID')[['movie_idx','Timestamp']].apply(lambda x: x.sort_values('Timestamp').movie_idx.agg(list)).to_dict()
        popular_movies = np.where(ratings.groupby('movie_idx').Rating.max()>=4)[0]
        vocab_size = max(movie_idx.values())+1
        return data, popular_movies, vocab_size
    
    def get_vocab_size(self):
        return self.vocab_size

    
    def __len__(self):
        return len(self.data)
    
    def get_negative_sample(self, user):
        movies_watched = self.data.get(user+1, [])
        negatives = self.popular_movies[~np.isin(self.popular_movies, movies_watched)]
        neg_sample = torch.tensor(np.random.choice(negatives, size=50, replace=False), dtype=torch.long)
        return neg_sample
        
    def __getitem__(self, user):
        seq = self.data.get(user+1, [])
        seq = seq[-(self.context_len+1):]
        pad_len = self.context_len+1 - len(seq)
        seq = [0] * pad_len + seq
        if self.split == 'train':
            tokens = torch.tensor(seq[-(self.context_len+1):-1], dtype=torch.long)
            labels = torch.full_like(tokens, fill_value=-100)
            mask_pos = np.random.rand(self.context_len) < self.mask_prob
            mask_pos[tokens==0] = False # don't mask padded tokens
            labels[mask_pos] = tokens[mask_pos]
            tokens[mask_pos]=0
            neg_sample = torch.tensor([])
        else:
            tokens = torch.tensor(seq[-self.context_len:], dtype=torch.long)
            labels = torch.full_like(tokens, fill_value=-100)
            mask_pos = torch.tensor([False]*(self.context_len-1)+[True])
            mask_pos[tokens==0] = False # don't mask padded tokens
            labels[mask_pos] = tokens[mask_pos]
            tokens[mask_pos]=0
            neg_sample = self.get_negative_sample(user)
        return {
            'user': user,
            'tokens': tokens,
            'labels': labels,
            'neg_sample': neg_sample
        }
