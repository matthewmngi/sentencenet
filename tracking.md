model - train loss - val loss
mlp + wavenet + sgd - 5.684243693584581 - 5.701639389819018
mlp + wavenet + adamw - 4.836368589899691 - 5.253106556217818
- at this point, sentences from the model start to become somewhat readable:
```
but then no sign we are no step allowed the second flow of my sources 
there 's recently are opposed mr. <unk> said <unk> does n't generates <unk> seven 
the <unk> familiar with daiwa laboratory is tested as regarded as lead of plaintiffs from american times while inadequate stock and 
the tokyo report fell N points in an elderly 
the charges includes N N east partnership has lost N N to N N N senior subordinated one-year buy-out preferred transport debt via <unk> at prudential-bache 
```
- at the same time, looking at the difference between train and val loss, we start to see some overfitting
rnn + adamw - x - x