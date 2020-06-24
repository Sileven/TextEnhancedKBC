import torch


class KBDataset(torch.utils.data.Dataset):
    def __init__(self, words, triples, triples_words, triple_len):
        assert len(words)==len(triples)
        assert len(triples)==len(triples_words)

        self._words=words
        self._triples=triples
        self._triples_words = triples_words
        self._triple_len = triple_len

    def __len__(self):
        return len(self._words)

    def __getitem__(self, idx):
        return self._words[idx], self._triples[idx], self._triples_words[idx], self._triple_len[idx]
