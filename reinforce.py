import time
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from misc.lang import Lang, pair2tensors
from misc.prep_data import MAX_LENGTH
from net.enc_dec import DEV_
from misc.plot import time_since, show_plot
# use for reinforcement learning
from nltk.translate import gleu_score
from torch.distributions import Categorical
import torch.nn.functional as F


def reinforce(input_tensor, target_tensor,
              target_sentence,  # used for calculating GLEU score
              encoder, decoder,
              output_lang,
              encoder_optimizer, decoder_optimizer,
              max_length=MAX_LENGTH,
              teacher_forcing_ratio=0.5,
              hypothesis_to_generate=20,
              baseline_reward=0.2):

    # Part 1: Generate k hypothesis sentences
    hyp_sents = []  # list of generated sentences
    hyp_probs = []  # their respecive log probabilities

    for k in range(hypothesis_to_generate):
        encoder_hidden = encoder.init_hidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=DEV_)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[Lang.SOS_token]], device=DEV_)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio \
            else False

        out_sent = []  # the kth generated sentence
        out_prob = 0  # and its respecive log probability

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input to decoder
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # topv, topi = decoder_output.topk(1)
                # out_sent.append(output_lang.idx2word[topi.item()])
                # out_prob += topv
                try:
                    m = Categorical(logits=decoder_output)
                    action = m.sample()

                    if action.cpu().item() == Lang.EOS_token:
                        break

                    out_sent.append(output_lang.idx2word[action.cpu().item()])
                    out_prob += decoder_output[0][action.cpu().item()]

                    decoder_input = target_tensor[di]  # Teacher forcing

                except Exception as e:
                    print(e)
                    breakpoint()
        else:
            # Without teacher forcing: use its own predictions as the next input to decoder
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # topv, topi = decoder_output.topk(1)
                # out_sent.append(output_lang.idx2word[topi.item()])
                # out_prob += topv
                try:
                    m = Categorical(logits=decoder_output)
                    action = m.sample()

                    if action.item() == Lang.EOS_token:
                        break

                    out_sent.append(output_lang.idx2word[action.cpu().item()])
                    out_prob += decoder_output[0][action.cpu().item()]

                    decoder_input = action.detach()

                except Exception as e:
                    print(e)
                    breakpoint()

                # FIXME Check this!
                # decoder_input = topi.squeeze().detach()  # detach from history as input
                # if decoder_input.item() == Lang.EOS_token:
                #    break

        hyp_sents.append(out_sent)
        hyp_probs.append(out_prob)

    hyp_probs = torch.stack(hyp_probs)  # turn list into tensor
    # FIXME normalize probability values
    hyp_probs = F.softmax(hyp_probs, dim=0)

    scores = []
    for k in range(hypothesis_to_generate):
        # Score the output sentence using GLEU
        score = gleu_score.sentence_gleu([target_sentence], hyp_sents[k])
        scores.append(score)

    scores = torch.tensor(scores, device=DEV_)

    reward = torch.sum(hyp_probs * scores)
    baseline = reward / hypothesis_to_generate  # TODO CHECK BASELINE

    loss = -torch.sum(torch.log(hyp_probs)) * (reward - baseline)
    print('loss {} - reward {} - baseline {}'.format(loss, reward, baseline))
    if loss < 1e-3:
        breakpoint()
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length  # TODO CHECK THIS


def reinforce_n_iters(encoder, decoder,
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

    print('Shuffling data...')
    training_pairs = [random.choice(pairs)
                      for i in range(n_iters)]

    print('Converting data to tensors...')
    training_tensors = [pair2tensors(
        pair, input_lang, output_lang) for pair in training_pairs]

    for iter in tqdm(range(1, n_iters + 1)):
        tensor_pair = training_tensors[iter - 1]
        sent_pair = training_pairs[iter - 1]
        input_tensor = tensor_pair[0]
        target_tensor = tensor_pair[1]
        target_sent = sent_pair[1].split()

        loss = reinforce(input_tensor, target_tensor, target_sent,
                         encoder, decoder, output_lang, encoder_optimizer, decoder_optimizer)

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
    }, 'encoder_reinf.pt')

    torch.save({
        'iters': n_iters,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': decoder.state_dict(),
        'loss': loss
    }, 'decoder_reinf.pt')

    show_plot(plot_losses)
    return encoder, decoder
