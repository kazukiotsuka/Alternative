# Alternative (Neura Voice)
original implementation of fast parallel speech signal generation from text with MFCC
(2018)

#### Idea & Architecture

- Text → (Location-based attention mechanism) →  MFCC   
- MFCC → (parallel recurrent network) → Speech Signal




#### Text to Mel
![FFTNet architecture](https://github.com/kazukiotsuka/Alternative/blob/master/notes/texttomel.png)

This model is based on Alex Glaves「Generating Sequences With Recurrent Neural Networks」

![FFTNet architecture](https://github.com/kazukiotsuka/Alternative/blob/master/notes/graves.png)
![FFTNet architecture](https://github.com/kazukiotsuka/Alternative/blob/master/notes/graves2.png)

![FFTNet architecture](https://github.com/kazukiotsuka/Alternative/blob/master/notes/graves3.png)

![FFTNet architecture](https://github.com/kazukiotsuka/Alternative/blob/master/notes/graves4.png)

![FFTNet architecture](https://github.com/kazukiotsuka/Alternative/blob/master/notes/graves5.png)


#### MFCC to Speech Signal

Parallel speech signal generation vocoder model (based on WaveRNN)

```python
    WaveRNN math::
        xt = [ct-1, ft-1, ct]  # input
        ut = σ(Ru ht-1 + Iu*xt + bu)  # update gate
        rt = σ(Rr ht-1 + Ir*xt + br)  # reset gate
        et = tanh(rt∘(Re ht-1) + Ie*xt + be)  # recurrent unit
        ht = ut∘ht-1 + (1-u)∘et  # next hidden state
        yc, yf = split(ht)  # coarse, fine
        P(ct) = softmax(O2 relu(O1 yc))  # coarse distribution
        P(ft) = softmax(O4 relu(O3 yf))  # fine distribution
```
