import torch
import torch.optim as optim

import random

from misc.prep_data import prepare_data, MAX_LENGTH
from net.enc_dec import DEV_, EncoderRNN, AttnDecoderRNN

from train import train_n_iters
from reinforce import reinforce_n_iters

from eval import evaluate_pairs

print("Preparing data...")
input_lang, output_lang, pairs = prepare_data('src', 'tgt', False)
print(random.choice(pairs))
# Keep some control data for evaluation
control_pairs = []
for i in range(150):
    control_pairs.append(random.choice(pairs))

# Train
print("training on {}".format(DEV_))

hidden_size = 256
# Make encoder and decoder
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(DEV_)
attn_decoder = AttnDecoderRNN(
    hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(DEV_)

# Setup optimizer
lr = 0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=lr)

# print('MLE Training for 500k iterations')
# train_n_iters(encoder, attn_decoder,
#               encoder_optimizer, decoder_optimizer,
#               500000, pairs, input_lang, output_lang, print_every=1000)

# Load from checkpoint
print('loading encoder...')
enc_chkpt = torch.load('encoder.pt')
encoder.load_state_dict(enc_chkpt['model_state_dict'])
encoder_optimizer.load_state_dict(enc_chkpt['optimizer_state_dict'])
iters_enc = enc_chkpt['iters']
loss_enc = enc_chkpt['loss']

print('loading decoder...')
dec_chkpt = torch.load('decoder.pt')
attn_decoder.load_state_dict(dec_chkpt['model_state_dict'])
# decoder_optimizer.load_state_dict(dec_chkpt['optimizer_state_dict'])
iters_dec = dec_chkpt['iters']
loss_dec = dec_chkpt['loss']

assert(iters_dec == iters_enc)
assert(loss_dec == loss_enc)

print('Reinforcement Learning for another 300k iterations')
reinforce_n_iters(encoder, attn_decoder,
                  encoder_optimizer, decoder_optimizer,
                  300000, pairs, input_lang, output_lang, print_every=1000)

evaluate_pairs(encoder, attn_decoder, control_pairs)
