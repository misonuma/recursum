## Unsupervised Abstractive Opinion Summarizationby Generating Sentences with Tree-Structured Topic Guidance
A code for "Unsupervised Abstractive Opinion Summarizationby Generating Sentences with Tree-Structured Topic Guidance
", TACL published in 2021.

Corresponding paper:
https://arxiv.org/abs/2106.08007

Masaru Isonuma, Juncihiro Mori, Danushka Bollegala, and Ichiro Sakata (The University of Tokyo, University of Liverpool)  

### Environment

Python 3.6

Run the following script to install required packages.
```
pip install -r requirements.txt

```

### Preprocessing

#### Yelp

- Download the raw Yelp data and run the pre-process script to create train, val and test directory following:  
https://github.com/sosuperic/MeanSum  

- Download the csv file containing reference summaries from:  
https://s3.us-east-2.amazonaws.com/unsup-sum/summaries_0-200_cleaned.csv  
(distributed in https://github.com/sosuperic/MeanSum)

- Run the following script:
```
python preprocess.py 
-data yelp 
-n_processes 32 # for multiprocessing
-path_train </path/to/preprocessed/train/dir>
-path_val </path/to/preprocessed/val/dir>
-path_test </path/to/preprocessed/test/dir>
-path_ref </path/to/reference/csv>
```

#### Amazon

Download the raw data and put them to `data/20news/` from  
https://github.com/akashgit/autoencoding_vi_for_topic_models/tree/master/data/20news_clean  
(The data is distributed in https://github.com/akashgit/autoencoding_vi_for_topic_models)


Run the following script:
```
python preprocess_20news.py -dir_data </dir/of/raw/data> -path_output </path/to/preprocessed/data>
```

### Training

Run the following script:

```
python train.py -gpu <index/of/gpu> -path_data </path/to/preprocessed/data> -dir_model <path/to/model/directory>
```

The trained parameters are saved in `dir_model`.  
The corpus in `dir_corpus` are used for calculating coherence score (NPMI).

### Evaluation

Run the following script:

```
python evaluate.py -gpu <index/of/gpu> -path_model <path/to/model/checkpoint> -dir_corpus <path/to/corpus>
```

The scores and topic frequent words are displayed in the console.  
You can also use our checkpoint in `model/bags/checkpoint_stable`.  
(Although the scores on this checkpoint slightly differ from the scores in the paper, the difference does not influence the claim of the paper.)  

### Acknowledgement

The module to calculate NPMI (`coherence.py`) is based on the code:  
https://github.com/jhlau/topic_interpretability
