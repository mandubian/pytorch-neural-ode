This repository is aimed at experimenting Different ideas with Neural-ODE in Pytorch

> You can contact me on twitter as [@mandubian](http://twitter.com/mandubian)

> All code is licensed under Apache 2.0 License


## NODE-Transformer

This project is a study about the NODE-Transformer, cross-breeding Transformer with Neural-ODE and based on [Facebook Fairseq Transformer](https://github.com/pytorch/fairseq) and [TorchDiffEq github](https://github.com/rtqichen/torchdiffeq).

An in-depth study can be found in this [notebook](node-transformer-fair/node-transformer-fair.ipynb) and you'll see that the main difference with usual Deep Learning studies is that it's not breaking any SOTA, it's not really successful or novel and worse, it's not at all ecological as it consumes lots of energy for not so good results. But I hope that as me, you will find those ideas and concepts enlightening and refreshing and finally worth the efforts.

----

**REQUEST FOR RESOURCES: If you like this topic and have GPU resources that you can share for free and want to help perform more studies on that idea, don't hesitate to contact me on Twitter @mandubian or Github, I'd be happy to consume your resources ;)**

----

### Implementation details

#### Hacking TorchDiffEq Neural-ODE

In this project, Pytorch is the framework used and Neural-ODE implementation is found in [torchdiffeq github](https://github.com/rtqichen/torchdiffeq).

TorchDiffEq Neural-ODE code is good for basic neural networks with one input and one output. But Transformer encoder/decoder is not really a basic neural network as attention network requires multiple inputs (Q/K/V) and different options.

Without going in details, we needed to extend TorchDiffEq code to manage multiple and optional parameters in `odeint_adjoint` and sub-functions. The code can be found [odeint_ext](https://github.com/mandubian/pytorch-neural-ode/tree/master/odeint_ext) and we'll see later if it's generic enough to be contribute it back to torchdiffeq project.


### Creating NODE-Transformer with fairseq

NODE-Transformer is just a new kind of Transformer as implemented in [FairSeq library](https://github.com/pytorch/fairseq).

So it was just implemented as a new kind of Transformer using FairSeq API, the [NODE-Transformer](https://github.com/mandubian/pytorch-neural-ode/blob/master/node-transformer-fair/node_transformer/node_transformer.py). Implementing it wasn't so complicated, the API is quite complete, you need to read some code to be sure about what to do but nothing crazy. _The code is still raw, not yet cleaned-up and polished so don't be surprised to find weird comments or remaining useless lines in a few places._

A custom [NODE-Trainer](https://github.com/mandubian/pytorch-neural-ode/blob/master/node-transformer-fair/node_transformer/node_trainer.py) was also required to integrate ODE function calls in reports. Maybe this part should be enhanced to make it more simply extensible

Here are the new options to manipulate the new kind of FairSeq NODE-Transformer:

```
    --arch node_transformer    
    --node-encoder
    --node-decoder
    --node-rtol 0.01
    --node-atol 0.01
    --node-ts [0.0, 1.0]
    --node-augment-dims 1
    --node-time-dependent
    --node-separated-decoder
```


