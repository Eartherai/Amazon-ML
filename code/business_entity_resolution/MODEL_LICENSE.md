# Model license and size

The submitted matcher is a LightGBM 4.7.0 gradient-boosted tree model saved as
`artifacts/model.txt`; it is not a language model or external checkpoint. The
installed LightGBM wheel declares `License-Expression: MIT` and includes
`licenses/LICENSE` beginning `The MIT License (MIT)`. The model contains 250
boosting iterations, far below the competition's 8-billion-parameter ceiling.

The model was trained solely from the supplied competition training records.
Frozen IDF vocabularies derive only from eligible training text. The bundled
generic transliteration cache derives from unlabeled competition test strings;
it contains no outside business identities.
