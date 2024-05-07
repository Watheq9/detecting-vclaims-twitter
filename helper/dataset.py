# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import configure as cf


class CLEFDataset(Dataset):
    def __init__(self, tweets, tweet_ids, vclaims, vlcaim_ids, labels, tokenizer, max_len,
                        ranks, lexical_similarities, semantic_similarities):
        self.labels = labels
        self.tweets = tweets
        self.vclaims = vclaims
        self.tweet_ids = [str(x) for x in tweet_ids]
        self.vlcaim_ids = [str(x) for x in vlcaim_ids] 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        if ranks is None:
            self.ranks = [1] * len(labels)
        else:
            self.ranks = ranks
        
        if lexical_similarities is None:
            self.lexical_similarities = [1] * len(labels)
            self.semantic_similarities = [1] * len(labels)
        else:
            self.lexical_similarities = lexical_similarities
            self.semantic_similarities = semantic_similarities

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        vclaim = str(self.vclaims[item])
        tweet_id = str(self.tweet_ids[item])
        vclaim_id = str(self.vlcaim_ids[item])
        label = self.labels[item]
        rank = self.ranks[item]
        lexical_similarity = self.lexical_similarities[item]
        semantic_similarity = self.semantic_similarities[item]


        encoding = self.tokenizer.encode_plus(
            tweet,
            vclaim,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            # pad_to_max_length=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            gb.TWEET_TEXT: tweet,
            gb.VCLAIM: vclaim,
            gb.VCLAIM_ID: vclaim_id,
            gb.TWEET_ID: tweet_id,
            gb.LABEL: torch.tensor(label, dtype=torch.long),
            gb.RANK: torch.tensor(rank, dtype=torch.long),
            gb.LEXICAL_SIMILARITY: lexical_similarity, 
            gb.SEMANTIC_SIMILARITY: semantic_similarity,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }



class CLEFDatasetWithTitle(Dataset):
    def __init__(self, tweets, tweet_ids, vclaims, vlcaim_ids, labels, tokenizer, max_len,
                        ranks, titles=None, lexical_similarities=None, semantic_similarities=None):
        self.labels = labels
        self.tweets = tweets
        self.vclaims = vclaims
        self.tweet_ids =  [str(x) for x in tweet_ids]  
        self.vlcaim_ids =  [str(x) for x in vlcaim_ids]  
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        if ranks is None:
            self.ranks = [1] * len(labels)
        else:
            self.ranks = ranks

        if titles is None:
            self.titles = " " * len(labels)
        else:
            self.titles = titles

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        vclaim = str(self.vclaims[item])
        tweet_id = str(self.tweet_ids[item])
        vclaim_id = str(self.vlcaim_ids[item])
        label = self.labels[item]
        rank = self.ranks[item]
        title = self.titles[item]

        query_title_encoding = self.tokenizer.encode_plus(tweet, title, 
            add_special_tokens=True, return_token_type_ids=False, padding="max_length",
            max_length=self.max_len,
            truncation=True, return_attention_mask=True, return_tensors="pt",)


        query_vclaim_encoding = self.tokenizer.encode_plus(tweet, vclaim, 
            add_special_tokens=True, return_token_type_ids=False, padding="max_length",
            max_length=self.max_len,
            truncation=True, return_attention_mask=True, return_tensors="pt",)

        return {
            gb.TWEET_TEXT: tweet,
            gb.VCLAIM: vclaim,
            gb.VCLAIM_ID: vclaim_id,
            gb.TWEET_ID: tweet_id,
            gb.LABEL: torch.tensor(label, dtype=torch.long),
            gb.RANK: torch.tensor(rank, dtype=torch.long),
            gb.QUERY_AND_TITLE_INPUT_IDS: query_title_encoding["input_ids"].flatten(),
            gb.QUERY_AND_TITLE_ATTENTION_MASK: query_title_encoding["attention_mask"].flatten(), 
            "input_ids": query_vclaim_encoding["input_ids"].flatten(),
            "attention_mask": query_vclaim_encoding["attention_mask"].flatten(),
        }



def create_data_loader(tweets, tweet_ids, vclaims, vlcaim_ids, labels, tokenizer, max_len, 
                        batch_size, num_workers=2, ranks=None, lexical_similarity=None, 
                        semantic_similarity=None):

    ds = CLEFDataset(tweets, tweet_ids, vclaims, vlcaim_ids, labels, tokenizer, max_len, 
                    ranks, lexical_similarity, semantic_similarity) 
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


def create_data_loader_with_title(tweets, tweet_ids, vclaims, vlcaim_ids, labels, tokenizer, max_len, 
                        batch_size, num_workers=2, ranks=None, lexical_similarity=None, 
                        semantic_similarity=None, titles=None):

    ds = CLEFDatasetWithTitle(tweets, tweet_ids, vclaims, vlcaim_ids, labels, tokenizer, max_len, 
                    ranks, lexical_similarities=lexical_similarity, 
                    semantic_similarities=semantic_similarity, titles=titles) 
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)




	