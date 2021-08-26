## Unsupervised Abstractive Opinion Summarizationby Generating Sentences with Tree-Structured Topic Guidance
A code for "Unsupervised Abstractive Opinion Summarizationby Generating Sentences with Tree-Structured Topic Guidance
", TACL to be published in 2021.

Corresponding paper:
https://arxiv.org/abs/2106.08007

Masaru Isonuma, Juncihiro Mori, Danushka Bollegala, and Ichiro Sakata (The University of Tokyo, University of Liverpool)  

---

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

- Run the following script (you can set any number to `n_processes` for multiprocessing):
```
python preprocess.py \
-data yelp \
-n_processes 16 \
-dir_train </path/to/train/dir> \
-dir_val </path/to/val/dir> \
-dir_test </path/to/test/dir> \
-path_ref </path/to/reference.csv>
```

#### Amazon

- Download the raw data from the following URL and unzip it:  
https://abrazinskas.s3-eu-west-1.amazonaws.com/downloads/projects/copycat/data/amazon.zip  
(distributed in https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer)  

- Download dev.csv and test.csv from:
https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/tree/master/gold_summs

- Run the following script :
```
python preprocess.py \
-data amazon \
-n_processes 16 \
-dir_train </path/to/train/dir> \
-path_val </path/to/dev.csv> \
-path_test </path/to/test.csv> 
```

The preprocessed data are saved in `data` by default.


### Training

- Run the following script:
```
python train.py \
-gpu <index/of/gpu> \
-data <"yelp"/or/"amazon"> \
-n_processes 16
```

The other arguments and default parameters are defined in `configure.py`.  
The trained parameters are saved in `model` by default.  


### Evaluation

- Run the following script (only single-gpu is available):  

```
python evaluate.py \
-gpu <index/of/gpu> \
-data <"yelp"/or/"amazon"> \
-n_processes 16
```

You need to set the same arguments as training except for `-gpu` and `-n_processes`.  
- You can also use our checkpoint in `model/yelp/recursum-stable` and `model/amazon/recursum-stable` as follows:  

```
python evaluate.py \
-gpu <index/of/gpu> \
-data <"yelp"/or/"amazon"> \
-n_processes 16 \
-stable
```


### Acknowledgement
