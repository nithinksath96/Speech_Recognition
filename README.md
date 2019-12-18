# Attention based End-End Speech-to-Text Deep Neural Networks

An attention based End-End Speech-to-Text Deep Neural Networks that learns to transcribe speech utterances to characters.
To this end, a combination of Recurrent Neural Networks (RNNs) / Convolutional Neural Networks (CNNs) and Dense Networks to design a system for speech to text transcription.


## Dataset

The Wall Street Journal (WSJ) dataset was used for this work. 

## Architecture

### LAS

In this work we use an encoder-decoder approach, called Listener and Speller respectively.

#### Listener
The Listener consists of a Pyramidal Bi-LSTM Network structure that takes in the given utterances and
compresses it to produce high-level representations for the Speller network.

#### Speller
The Speller takes in the high-level feature output from the Listener network and uses it to compute a
probability distribution over sequences of characters using the attention mechanism.

#### Attention
Attention intuitively can be understood as trying to learn a mapping from a word vector to some areas of
the utterance map. The Listener produces a high-level representation of the given utterance and the Speller
uses parts of the representation (produced from the Listener) to predict the next word in the sequence.

### LAS VARIANT-1

In this work, instead of using one projection in the attention module, we could instead take two projections and use them as an Attention Key and an Attention Value.
Your encoder network over the utterance features should produce two outputs, an attention value and a
key and your decoder network over the transcripts will produce an attention query. We are calling the dot
product between that query and the key the energy of the attention. Feed that energy into a Softmax,
and use that Softmax distribution as a mask to take a weighted sum from the attention value (apply the
attention mask on the values from the encoder). That is now called attention context, which is fed back into
your transcript network.

#### Teacher Forcing

One problem you will encounter in this setting is the difference of training time and evaluation time: at
test time you pass in the generated character/word from your model, when your network is used to having
perfect labels passed in during training. One way to help your network be better at accounting for this noise
is to actually pass in your generated chars/words during training, rather than the true chars/words, with
some probability. This is known as **Teacher Forcing**.

#### Gumbel Noise

Another problem you will be facing is that given a particular state as input to your model, the model will
always generate the same next state output, this is because once trained, the model will give a fixed set of
outputs for a given input state with no randomness. To introduce randomness in your prediction, you will
want to add some noise into your prediction (only during generation time) specifically the **Gumbel noise**.


#### Evaluation
Performance is evaluated using CER - character error rate (edit distance).

#### Results
The given model achieves CER of 11.33 on WSJ dataset.

#### References

Listen, Attend and Spell:https://arxiv.org/pdf/1508.01211.pdf
