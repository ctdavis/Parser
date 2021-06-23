Using an autoencoder that is constrained by being required to produce a binary tree on the decoding end, a constituency parser can be learned without needing explicit examples of syntax trees. This is interesting because humans appear to have intuitions about grammar that they are not taught directly (an idea known as poverty of the stimulus) and giving a machine access to syntax without needing annotations is promising in terms of costs and applications.

In order to run the parser, simply execute the following line:

    python parser.py
    
Without any changes to the code, training currently takes up around 5 GB, but this can be nearly cut in half if you comment out the consistency and clustering losses. Training the model up to around 90% reconstruction accuracy can take a few hours (or after the observation of about 10,000+ sentences). While the syntax trees that the model produces are promising, they often do not match my own intuitions of sentence structures.
