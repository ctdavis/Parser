Using an autoencoder, one can parse sentences by using some encoder (currently, I'm using a residual convolutional network) and then
decoding using a set of fully connected layers that decide whether to branch at a given node as well as determine states for two child nodes if the decision is to branch. The decoder recursively builds a syntax tree until each branch has decided to terminate or the decoder runs out of a certain number of nodes it is allotted based on the length of the input sentence. In order to cope with OOV words, an (I think simplified) implementation of CopyNet is used in order to let the parser generate words it has in its vocabulary or to select indices from the input sentence.

This parser learns grammar without ever needing to see an example of a syntax tree (correct or otherwise) which makes it interesting for two main reasons: first, this detail fits an observation about human language learning which is that humans have consistent intuitions about compositional structure in language without necessarily being given explicit instruction on grammar; second, in practical applications where one might need access to explicit syntactic structure, there is no need to annotate data to achieve such a goal.

The following sentences are from the Twitter sentiment dataset. As far as the first sentence, I wouldn't say the parse is exactly how I would do it, but the fact that the two clauses\* in the tweet are on separate branches and the verb phrase (VP) and noun phrase (NP) on the left branch are correctly separated\*\* makes this parser a promising candidate for general use in NLP.

(after preprocessing) unable to access the website . will try again .

(after unknown words are replaced) \<unk> to \<unk> the \<unk> . will try again .

<pre>                                                                               
                      ___________________|______________________                    
                                                                                   
           __________|____________                        ______|____________       
                                  |                                          |     
    ______|____                   |                  ____|______             |      
   |                                                            |            |      
   |       ____|_____         ____|______        ___|____       |       _____|___   
'unable' 'to'     'access' 'the'     'website' '.'     'will' 'try' 'again'     '.'
   |      |          |       |           |      |        |      |      |         |  
  ...    ...        ...     ...         ...    ...      ...    ...    ...       ...
</pre>

(after preprocessing) the magic eight ball has never steered me wrong :)

(after unknown words are replaced) the \<unk> \<unk> \<unk> has never \<unk> me \<unk> :)

<pre>                                                                                     
                    _______________________|___________________                       
                   |                                                                 
                   |                               ____________|___________           
                                                                           |         
        ___________|___________             ______|_____                   |          
                                           |                                         
   ____|_____             _____|____       |       _____|______        ____|_____     
'the'     'magic'     'eight'     'ball' 'has' 'never'     'steered' 'me'     'wrong'
  |          |           |          |      |      |            |      |          |    
 ...        ...         ...        ...    ...    ...          ...    ...        ...  
</pre>

\* a clause is a complete proposition - in English, at least a subject and a predicate, although colloquially it is sometimes grammatical to drop the subject, as is demonstrated here
\*\* the left and right sides are actually all a VP with an object - in this case "the website"
