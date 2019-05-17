# fightin_words
A packaged Python 3 implementation of the weighted log-odds-ratio method from Monroe et al., 2008. This code was based on Jack Hessel's original Python 2 script. See references section for more details.  

Dependencies
---
scikit-learn

Installing
---
Clone the repository, and from the directory under which root folder is located, run:
```
pip install fightin_words
```

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

Known Issues
---
Due to apparent limitations within the backend of the dependencies, intermediate matricial operations take up memory at a very high spatial complexity, meaning that massive amounts of memory may be required to successfully complete the execution of code, depending on the size of the dataset being used.  

References
---
J. Hessel, “Fightin’ Words,” GitHub, 2016. [Online]. Available: https://github.com/jmhessel/FightingWords. [Accessed: 26-Apr-2019].

B. L. Monroe et al., “Fightin’ Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflic,” Polit. Anal., vol. 16, no. 4, pp. 372–403, 2008.

Citation
---
P. H. M. Wigderowitz, “fightin_words,” GitHub, 2019. [Online]. Available: https://github.com/Wigder/fightin_words. [Accessed: XX-XXX-XXXX]. doi:

License
---
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
