#  main script
import random

from misc.prep_data import prepare_data, MAX_LENGTH
from net.enc_dec import DEV_, EncoderRNN, AttnDecoderRNN

from train import train_n_iters
from reinforce import reinforce_n_iters

from eval import evaluate_pairs

print("Preparing data...")
input_lang, output_lang, pairs = prepare_data('src', 'tgt', False)
print(random.choice(pairs))

# Train
print("training on {}".format(DEV_))
control_pairs = []
for i in range(150):
    control_pairs.append(random.choice(pairs))


hidden_size = 256


encoder2 = EncoderRNN(input_lang.n_words, hidden_size).to(DEV_)
attn_decoder2 = AttnDecoderRNN(
    hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(DEV_)

print('MLE Training for 500k iterations')
train_n_iters(encoder2, attn_decoder2, 500000, pairs, input_lang, output_lang, print_every=1000)

print('Reinforcement Learning for another 300k iterations')
reinforce_n_iters(encoder2, attn_decoder2, 300000, pairs, input_lang, output_lang, print_every=1000)

evaluate_pairs(encoder2, attn_decoder2, control_pairs)
