# Dataset Provenance

Vendored MCE datasets in this directory:

- `finer/`: copied from `metaevo-ai/mce-artifact/env/finer/data`
- `uspto/`: copied from `metaevo-ai/mce-artifact/env/uspto/data`
- `symptom_diagnosis/`: copied from `metaevo-ai/mce-artifact/env/symptom_diagnosis/data`
- `crime_prediction/`: copied from `metaevo-ai/mce-artifact/env/crime_prediction/data`
- `aegis2/`: copied from `metaevo-ai/mce-artifact/env/aegis2/data`

Kept OOD datasets loaded at runtime from HuggingFace:

- `AGNews`: `fancyzhx/ag_news`
- `GoEmotions`: `google-research-datasets/go_emotions`, config `simplified`
- `Banking77`: `banking77`
- `FinancialPhraseBank`: `financial_phrasebank`, config `sentences_allagree`
- `SciCite`: `allenai/scicite`, revision `refs/convert/parquet`
- `TweetEval_hate`: `tweet_eval`, config `hate`
- `Amazon5`: `SetFit/amazon_reviews_multi_en`
- `SciTail`: `allenai/scitail`, config `snli_format`
