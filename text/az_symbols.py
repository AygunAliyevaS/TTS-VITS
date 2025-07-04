import torch

class AzerbaijaniPhonemes:
    def __init__(self, device='cuda'):
        self.device = device
        self.xlsr = torch.hub.load('facebookresearch/fairseq', 'xlsr_300m').to(device)
        self.xlsr.eval()
        
        self.phonemes = [
            'a', 'æ', 'b', 'tʃ', 'd', 'e', 'f', 'g', 'ɣ', 'h', 'x',
            'i', 'dʒ', 'k', 'l', 'm', 'n', 'o', 'œ', 'p', 'q', 'ɾ',
            's', 'ʃ', 't', 'u', 'y', 'v', 'j', 'z', 'ʒ', 'ʔ'
        ]

    def get_phoneme_ids(self, text):
        return [self.phonemes.index(c) for c in text if c in self.phonemes]
