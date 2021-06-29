Using an autoencoder that is constrained by being required to produce a binary tree on the decoding end, a constituency parser can be learned without needing explicit examples of syntax trees. This is interesting because humans appear to have intuitions about grammar that they are not taught directly (an idea known as poverty of the stimulus) and giving a machine access to syntax without needing annotations is promising in terms of costs and applications.

In order to train the parser, execute the following line:

    python parser.py
