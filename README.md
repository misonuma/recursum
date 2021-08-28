## Unsupervised Abstractive Opinion Summarizationby Generating Sentences with Tree-Structured Topic Guidance
- A code for "Unsupervised Abstractive Opinion Summarizationby Generating Sentences with Tree-Structured Topic Guidance", TACL to be published in 2021.

  - Corresponding paper: https://arxiv.org/abs/2106.08007

  - Masaru Isonuma, Juncihiro Mori, Danushka Bollegala, and Ichiro Sakata (The University of Tokyo, University of Liverpool)  

- Output examples for all reviews in dev/test splits are avalable at `output/yelp` and `output/amazon`.

---

### Environment

- Python 3.6

- Run the following script to install required packages.
```
pip install -r requirements.txt
```


### Preprocessing

#### Yelp (based on MeanSum@ICML2019, http://proceedings.mlr.press/v97/chu19b/chu19b.pdf)

- Download the raw Yelp data and run the pre-process script to create train, val and test directory following:  
https://github.com/sosuperic/MeanSum  

- Download the csv file containing reference summaries from:  
https://s3.us-east-2.amazonaws.com/unsup-sum/summaries_0-200_cleaned.csv  
(distributed in https://github.com/sosuperic/MeanSum)

- Run the following script (you can set any number to `n_processes` for multiprocessing):
```
python preprocess_yelp.py \
-n_processes 16 \
-dir_train </path/to/train/dir> \
-dir_val </path/to/val/dir> \
-dir_test </path/to/test/dir> \
-path_ref </path/to/reference.csv>
```

The preprocessed data are saved in `data/yelp` by default.

#### Amazon (based on Copycat@ACL2020, https://aclanthology.org/2020.acl-main.461/)

- Download the raw data from the following URL and unzip it:  
https://abrazinskas.s3-eu-west-1.amazonaws.com/downloads/projects/copycat/data/amazon.zip  
(distributed in https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer)  

- Download dev.csv and test.csv from:
https://github.com/abrazinskas/Copycat-abstractive-opinion-summarizer/tree/master/gold_summs

- Run the following script :
```
python preprocess_amazon.py \
-n_processes 16 \
-dir_train </path/to/train/dir> \
-path_dev </path/to/dev.csv> \
-path_test </path/to/test.csv> 
```

The preprocessed data are saved in `data/amazon` by default.


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

You need to set the same arguments as those used for training except for `-gpu` and `-n_processes`.  
You can also use our checkpoint in `model/yelp/recursum-stable` and `model/amazon/recursum-stable` by adding `-stable` flag.

### Acknowledgement

We would like to acknowledge MeanSum & Copycat authors for providing reference summaries of Yelp & Amazon reviews.  

For calculating ROUGE scores, we use the same scripts used in MeanSum (`evaluation`).  
https://github.com/sosuperic/MeanSum/tree/master/evaluation
