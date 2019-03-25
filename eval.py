
import torch
import random
from net.enc_dec import DEV_
from misc.lang import Lang, sentence2tensor
from misc.plot import show_attention
from misc.prep_data import MAX_LENGTH


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    """
    Mostly the same as training.
    There are no targets:

        (1) - feed the decoder’s predictions
        back to itself at each step

        (2) - each time it predicts a word add it to the output string
              stop if EOS token is predicted

    Store the decoder’s attention outputs for display later
    """
    with torch.no_grad():
        input_tensor = sentence2tensor(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=DEV_)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[Lang.SOS_token]], device=DEV_)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == Lang.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.idx2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_pairs(encoder, decoder, pairs):
    for pair in pairs:
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluate_randomly(pairs, encoder, decoder, n=10):
    """
    evaluate random sentences from the training set
    print out to make some subjective quality judgements
    """
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def eval_and_show_attention(input_sentence, encoder, decoder):
    output_words, attentions = evaluate(encoder, decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)
