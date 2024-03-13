# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 19:10:13 2024

@author: 729sh
"""
import torch

class Text_handler():
    def __init__(self, path_to_text):
        self.path = path_to_text
    
    def load_text(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.unique_chars = sorted(set(self.text))
        
    def load_encoder(self):
        string_to_int = {ch:i for i,ch in enumerate(self.unique_chars)}
        self.encode = lambda s: [string_to_int[c] for c in s]
    
    def load_decoder(self):
        int_to_string = {i:ch for i,ch in enumerate(self.unique_chars)}
        self.decode = lambda l: "".join([int_to_string[i] for i in l])
        
    def as_tensor(self):
        self.text_tensor = torch.tensor(self.encode(self.text), dtype = torch.long)
        
    def prepare_text(self):
        self.load_text()
        self.load_encoder()
        self.load_decoder()
        self.as_tensor()
    
    def encode(self, s):
        return self.encode(s)
    
    def decode(self, i):
        return self.decode(i)
    
    def split(self, fraction):
        n = int(fraction*len(self.text_tensor))
        train = self.text_tensor[:n]
        val = self.text_tensor[n:]
        return train, val
        
        
if __name__ == "__main__":
    path = "../data/wizard_of_oz.txt"
    prep = Text_handler(path)
    prep.prepare_text()
    text = "shutter"
    encoding = prep.encode(text)
    
    decoding = prep.decode(encoding)
    
    print(text, encoding, decoding)
    