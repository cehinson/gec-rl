
from __future__ import unicode_literals, print_function, division
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from misc.lang import Lang, pair2tensors
from misc.prep_data import MAX_LENGTH
from misc.plot import show_plot, time_since
from net.enc_dec import DEV_


def train(input_tensor, target_tensor,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion,
          max_length=MAX_LENGTH,
          teacher_forcing_ratio=0.5):

    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEV_)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden
        )
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[Lang.SOS_token]], device=DEV_)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio \
        else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input to decoder
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input to decoder
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == Lang.EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def train_n_iters(encoder, decoder,
                  encoder_optimizer, decoder_optimizer, n_iters,
                  pairs, input_lang, output_lang,
                  print_every=1000, plot_every=100):
    """
    The whole training process looks like this :
        -- Start a timer
        -- Initialize optimizers and criterion
        -- Create set of training pairs
        -- Start empty losses array for plotting
    """

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    print('Shuffling training data')
    training_pairs = [random.choice(pairs)
                      for i in range(n_iters)]

    print('Converting training data to tensors')
    training_tensors = [pair2tensors(
        pair, input_lang, output_lang) for pair in training_pairs]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        tensor_pair = training_tensors[iter - 1]
        input_tensor = tensor_pair[0]
        target_tensor = tensor_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    torch.save({
        'iters': n_iters,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': encoder_optimizer.state_dict(),
        'loss': loss
    }, 'encoder.pt')

    torch.save({
        'iters': n_iters,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': decoder_optimizer.state_dict(),
        'loss': loss
    }, 'decoder.pt')

    show_plot(plot_losses)

    return encoder, decoder
