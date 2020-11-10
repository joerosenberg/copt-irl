This repository contains the code produced for my master's dissertation, "Learning Shaped Reward via Adversarial Imitation Learning for Combinatorial Optimisation".

In my dissertation, I produced and analysed a method for learning to route circuits from exploration and demonstration. Unfortunately, I have not yet received permission to share
the circuit routing code from the company I worked with, but all of the code on the ML/RL and analysis side are included in this repo.

The model I created is primarily inspired by the Transformer-based model from ["Attention, Learn to Solve Routing Problems!" (Kool et al. 2019)](https://arxiv.org/abs/1803.08475),
while the adversarial training method is inspired by ["Learning Robust Rewards with Adversarial Inverse Reinforcement Learning" (Fu et al. 2018)](https://arxiv.org/abs/1710.11248).

Further details of the model are included in Section 6 of my dissertation, which I have included in the repo. The structure of the repo is described below.

Model code
==========

-   `irlco.pointer_transformer`: Specifies the policy and reward-shaping
    models as PyTorch modules.

-   `irlco.pointer_multihead`: Modified versions of the multi-head
    attention modules from the PyTorch library that return attention
    scores rather than attention weights. This is necessary for the
    final layer of the decoder in the policy model.

-   `irlco.routing.reward`: Methods associated with the reward model.
    Includes a method for computing shaping terms directly from
    environment states using the reward-shaping net, and a method for
    calculating rewards from shaping terms.

-   `irlco.routing.policy`: Methods associated with the policy model.
    Includes methods for generating action sequences using greedy
    search, beam search and sampling.

Environment
===========

-   `irlco.routing.env`: Implements the circuit routing MDP as an OpenAI
    Gym environment. Also contains a method for calculating terminal
    rewards from solution measures and successes, and a method for
    generating problem instances with at least 30 units of clearance
    between nodes.

Training
========

-   `irlco.routing.train`: Script that trains models for the circuit
    routing problem. Includes configurable parameters for changing model
    architecture, enabling learning shaped rewards, as well as other
    training hyperparameters. Also includes remote logging of model
    configuration and training performance through the Weights & Biases
    API, and a method for saving and loading expert data from pickle
    files to decrease startup time.

Data
====

-   `irlco.routing.data.generate_training_data`: Methods for generating
    and saving brute-forced solutions to circuit routing problems. The
    quantity and type of data to generate can be specified in a YAML
    config file, which then serves as a manifest for the data after
    generation. Two example config files are supplied.

-   `irlco.routing.data`: Implements a class for loading generated data
    according to a YAML config file, and allows the loaded data to be
    accessed in batches. In practice, instances of these classes are
    pickled by the training script after loading for the first time, as
    parsing the data from text files takes a long time.

Experiments
===========

-   `irlco.routing.copt_analysis`: Used to find the number of solutions
    and ranges of measures in Chapter 3. The scripts used to create the
    plots are in the `irlco/routing/copt_analysis` folder.

-   `irlco.routing.beam_search_experiment`: Used to evaluate beam search
    performance of the trained models in the second experiment.

-   `irlco.routing.active_search_experiment`: Used to evaluate active
    search performance in the third experiment.


