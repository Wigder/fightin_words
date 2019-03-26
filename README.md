# fightin_words

Usage
---
Uninformed example:
    
```python
from sklearn.feature_extraction.text import CountVectorizer
from fightin_words import weighted_log_odds_dirichlet

doc1 = ["the quick brown fox jumps over the lazy pig"]
doc2 = ["the lazy purple pig jumps over the lazier donkey"]

out = weighted_log_odds_dirichlet(doc1, doc2, prior=.05, cv=CountVectorizer())
expected_out = [("donkey", -0.6894227886807968), ("lazier", -0.6894227886807968), ("purple", -0.6894227886807968),
                ("jumps", 0.0), ("lazy", 0.0), ("over", 0.0), ("pig", 0.0), ("the", 0.0), ("brown", 0.6894227886807968),
                ("fox", 0.6894227886807968), ("quick", 0.6894227886807968)]

# print(dict(out) == dict(expected_out))
```

Informed example:

```python
from collections import Counter

from fightin_words import weighted_log_odds_dirichlet
from sklearn.feature_extraction.text import CountVectorizer

doc1 = ["the quick brown fox jumps over the lazy pig"]
doc2 = ["the lazy purple pig jumps over the " + (1000 * "very ") + "sleepy donkey"]
uninformed_prior = .05

word_count = dict(Counter((doc1[0] + " " + doc2[0]).split()).most_common())

shrinking_target = "very"  # As an example, we intend to shrink the z-score of "very".
priors = {shrinking_target: 1.0 / word_count[shrinking_target]}
for w in word_count:
    if w not in priors:
        priors[w] = uninformed_prior

prior = []
cv_vocab = {}
for w, p in priors.items():
    prior.append(p)
    cv_vocab[w] = len(prior) - 1

uninformed_out = weighted_log_odds_dirichlet(doc1, doc2, prior=uninformed_prior, cv=CountVectorizer())
informed_out = weighted_log_odds_dirichlet(doc1, doc2, prior=prior, cv=CountVectorizer(vocabulary=cv_vocab))

# expected_uninformed_out = [('very', -2.2144429606255946), ('donkey', 0.3528670051726073),
#                            ('purple', 0.3528670051726073), ('sleepy', 0.3528670051726073),
#                            ('brown', 1.7074955767167836), ('fox', 1.7074955767167836), ('quick', 1.7074955767167836),
#                            ('jumps', 3.4564380059373603), ('lazy', 3.4564380059373603), ('over', 3.4564380059373603),
#                            ('pig', 3.4564380059373603), ('the', 4.954523357120953)]
# expected_informed_out = [('very', -0.4368846171934332), ('purple', 0.3539801999989518),
#                          ('sleepy', 0.3539801999989518), ('donkey', 0.3539801999989518), ('quick', 1.7087406001344665),
#                          ('brown', 1.7087406001344665), ('fox', 1.7087406001344665), ('jumps', 3.4605672465974453),
#                          ('over', 3.4605672465974453), ('lazy', 3.4605672465974453), ('pig', 3.4605672465974453),
#                          ('the', 4.961066225017743)]
# 
# print(dict(uninformed_out) == dict(expected_uninformed_out))
# print(dict(informed_out) == dict(expected_informed_out))
```
