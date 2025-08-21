Code and data to support the following paper:

David Bamman, Sabrina Baur, Mackenzie Hanh Cramer, Anna Ho and Tom
 McEnaney (2025), "Measuring the Stories in Songs" (currently in submission), [pdf](https://osf.io/preprints/socarxiv/5ys9c_v1).

# Data

* `data/annotations.tsv` -- all narrativity annotations, paired with azlyrics URL.
* `data/billboard_hot_100.tsv` -- Billboard Hot 100 lists 1960-2024.
* `data/billboard_subcharts.tsv` -- Billboard Top {Country, Pop, Rap, R&B, R&B/Hiphop and Rock} subcharts.
* `data/grammy_subawards.tsv` -- Grammy award nominees for Best {Country, R&B, Rap, and Rock} song, paired with a random song from the same album.
* `data/*jsonl` files are the original files we use for training and prediction, but with the lyrics (which we cannot re-publish) removed.  Each song includes a link to lyrics on azlyrics.com.


# Install 

```
conda create --prefix /path/to/conda/songnarrativity python=3.12
conda activate /path/to/conda/songnarrativity
pip install transformers torch scipy protobuf tiktoken sentencepiece optuna booknlp scikit-learn
python -m spacy download en_core_web_sm
```


# Train and evaluate models

Masked LMs

```
cd scripts

for MODEL in bert deberta roberta modern-bert
do

python bert_narrativity_single.py --trainFile ../data/annotated_data.jsonl --mode train --agent_model ../logs/agents.${MODEL}_1960_2024.model --base ${MODEL} --task agents --device cuda:0 > ../logs/agents.${MODEL}.log 2>&1

python bert_narrativity_single.py --trainFile ../data/annotated_data.jsonl --mode train --event_model ../logs/events.${MODEL}_1960_2024.model --base ${MODEL} --task events --device cuda:0  > ../logs/events.${MODEL}.log 2>&1

python bert_narrativity_single.py --trainFile ../data/annotated_data.jsonl --mode train --world_model ../logs/world.${MODEL}_1960_2024.model --base ${MODEL} --task world --device cuda:0  > ../logs/world.${MODEL}.log 2>&1

python bert_narrativity_single.py --trainFile ../data/annotated_data.jsonl --mode evaluate --agent_model ../logs/agents.${MODEL}_1960_2024.model --event_model ../logs/events.${MODEL}_1960_2024.model --world_model ../logs/world.${MODEL}_1960_2024.model --base ${MODEL} --predictionFile ../logs/all.out.${MODEL}.preds --task all --device cuda:0

python bootstrap_spearman.py ../logs/all.out.${MODEL}.preds

done

```

Featurized

```
for features in imemy minimal bow+pos+animacy+concrete+imemy pos+animacy+concrete+imemy
do

python featurized_narrativity_sklearn.py --trainFile ../data/annotated_data.jsonl --mode train --predictionFile ../logs/logreg.${features}.preds --concreteness_file ../data/Concreteness_ratings_Brysbaert_et_al_BRM.txt --booknlp_output_folder ../booknlp_output/ --feature_set ${features} > ../logs/logreg.${features}.log 2>&1

done
```


# Download trained RoBERTa models

```
wget -P logs/ http://yosemite.ischool.berkeley.edu/david/song_narrativity/agents.roberta_1960_2024.model
wget -P logs/ http://yosemite.ischool.berkeley.edu/david/song_narrativity/events.roberta_1960_2024.model
wget -P logs/ http://yosemite.ischool.berkeley.edu/david/song_narrativity/world.roberta_1960_2024.model
```

# Predict


Using RoBERTa (as the best-performing model above) to predict narrativity for all songs in the Billboard Hot 100, subcharts, and grammy nominees.

```
python bert_narrativity_single.py --trainFile ../data/billboard100.jsonl --mode predict --agent_model ../logs/agents.roberta_1960_2024.model --event_model ../logs/events.roberta_1960_2024.model --world_model ../logs/world.roberta_1960_2024.model --base roberta --device cuda:2 --predictionFile ../data/all.out.roberta.billboard100.preds

python bert_narrativity_single.py --trainFile ../data/billboard_subcharts.jsonl --mode predict --agent_model ../logs/agents.roberta_1960_2024.model --event_model ../logs/events.roberta_1960_2024.model --world_model ../logs/world.roberta_1960_2024.model --base roberta --device cuda:2 --predictionFile ../data/all.out.roberta.billboard_subcharts.preds

python bert_narrativity_single.py --trainFile ../data/grammy_subawards.jsonl  --mode predict --agent_model ../logs/agents.roberta_1960_2024.model --event_model ../logs/events.roberta_1960_2024.model --world_model ../logs/world.roberta_1960_2024.model --base roberta --device cuda:2 --predictionFile ../data/all.out.roberta.grammy_subawards.preds
``` 

# Generate plots, tables, statistics


Correlation over time between year and mean narrativity

```
python scripts/pop_correlation.py data/all.out.roberta.billboard100.preds data/billboard_hot_100.tsv
```


Distribution of annotations for agents, events, world-building

```
cd plots
Rscript dist.r 
```

billboard\_hot100\_year.pdf

```
python scripts/calc_time.py data/all.out.roberta.billboard100.preds data/billboard_hot_100.tsv > data/billboard.hot100.yearly.data
cd plots
Rscript billboard_hot100_time.r
```

Narrativity by genre table

```
 python scripts/calc_genre.py data/all.out.roberta.billboard100.preds data/billboard_hot_100.tsv
```

billboard\_subcharts\_year.pdf

```
python scripts/calc_subgenre.py data/all.out.roberta.billboard_subcharts.preds data/billboard_subcharts.tsv > data/billboard.subcharts.yearly.data
cd plots
Rscript subcharts.r
```

hiphop\_rap.pdf

```
python scripts/genre_proportion_by_year.py data/billboard_hot_100.tsv > data/hiphop_rap_country.proportion.txt
cd plots
Rscript genre.r
```

Narrativity by award category table.

```
python scripts/test_anno_grammy.py data/annotations.tsv

python scripts/test_grammy.py data/grammy_subawards.tsv data/all.out.roberta.grammy_subawards.preds 
```

billboard\_hot100\_manual\_year.pdf

```
python scripts/calc_time_manual.py data/annotations.tsv > data/billboard.hot100_manual.yearly.data
cd plots
Rscript billboard_hot100_time_manual.r 
```