# PRISM

This is an implementation of the experiments in paper:

Yixin Yang, Ming Wen, Yuting Zhang, Kaixuan Luo and Hai Jin, "Exploring and Exploiting Code Representation Learning for Static Bug Detectors"

## Requirements

Tested on ubuntu 18.04:

- Python 3.8
- Tensorflow 2.4.0
- Keras 2.4.3
- Pytorch 1.8.1
- Gensim 3.8.3
- Numpy 1.19.0



## Usage

Link of dataset and pretrained embedding models:

https://drive.google.com/drive/folders/1_yb9dSjIkq4Q8NbwfYfGmwh6TY462CFg?usp=sharing

Download two datasets and pretrained models from the link and save to YOURPATH(YOURPATH/dataset, YOURPATH/pre_trained_model).



### Evaluating CRL models for bug detection (RQ1)

```
git clone https://github.com/SAS-HUST2022/PRISM.git

cd PRISM

mv YOURPATH/dataset/dataset1 ./all_embedding/pickle_object/spotbugs

mv YOURPATH/pre_trained_model/model4data1 ./all_embedding/pre_train_def/spotbugs

cd ./all_embedding/Word2vec(fasttext/Glove/Elmo/Bert)/src_code

python -u src_word2vec.py(src_fasttext.py/src_glove.py/src_elmo.py/src_bert.py) --cls spot_bin --code src --fold 11 --retrain False --split_test True --neural lstm (--bert_model bert_seq)

cd spot_bin(spot_mul)/src_w2v_lstm (results are saved in best_score_record)
```

--cls:  the kind of data, "spot_bin" denotes the binary bug classification, "spot_mul" denotes the multi-class bug classification.

--code:  the kind of code, it is fixed to "src".

--fold:  the number of fold for conducting cross validation, it is fixed to "11", which means the dataset is split to 9:1:1.

--retrain: whether to retrain the pretrained embeddding model, as we release the trained CRL models, it is set to "False". You can set it to "True" if you want to retrain the CRL models.

--split_test:  whether to split the test set, it is fixed to "True".

--neural:  the detecting neural network, it can be set to lstm/blstm/gru/bgru/textcnn/mlp for different learning models.

--bert_model:  the pretrained bert model, it is specified only when you run "src_bert.py", it can be set to "bert_token/roberta_token/codebert_token" for token-level experiments, and can be set to "bert_seq/roberta_seq/codebert_seq" for sequence-level experiments (when you set "--bert_model" for sequence-level experiments, "--neural" needs to be set to "lr").



### Evaluating CRL models for differentiating actionable and unactionable warnings (RQ2)

```
cd PRISM && cd all_embedding

mv YOURPATH/dataset/dataset2 ./app/data

mv YOURPATH/pre_trained_model/model4data2 ./app/model/src_top_20

cd ./app

python sort_bug.py --filter 0 --balance 1 --train 1 ---method embedding --code src --embed w2v --detect gru 

cd ./score/w2v_gru/ (results are saved in best_score_record)
```

--filter:  whether to filter the cross-method bug

--balance:  whether to balance the dataset, it is fixed to "1".

--train:  whether to train detecting model, it is fixed to "1".

--method:  run our embedding model or baseline model. 

--code:  the kind of code, it is fixed to "src".

--embed:  the kind of CRL models, it can be set to w2v/fasttext/glove/elmo when evaluating Word2Vec/FastText/GLoVe/ELMo. When evaluating pre-trained models, it can be set to bert_seq/roberta_seq/codebert_seq when evaluating sequence-level BERT/RoBERTa/CodeBERT, and bert_token/roberta_token/codebert_token when evaluating token-level BERT/RoBERTa/CodeBERT.

--detect:  the kind of detecting NN models, it can be set to lstm/blstm/gru/bgru/textcnn. It is set to "lr" when "--embed" is set to bert_seq/roberta_seq/codebert_seq.



## Evaluating PRISM and baselines (RQ3)

- running PRISM

```
cd PRISM && cd all_embedding && cd Hybrid && cd new

python voting.py --vote True --embed group3

cd ../../app/vote_score/(vote_group3/vote_group5/vote_group7/vote_group9)/ (to see results, results are saved in "hard_vote_results" and "soft_vote_results")
```

--vote: whether to use the Majority vote mechanism.

--embed: the experimental setting for PRISM, "w2v_bgru" denotes the best CRL model,  "group3", "group5", "group7", "group9" denote to combine the top 3, top 5, top 7, top 9 CRL models together to conduct experiments.



- running baseline1: HWP 	

```
cd ./app_embedding/app/

python sort_bug.py --filter 0 --balance 1 --train 0 --method hwp --code src 
```

- running baseline2: the "Golden features" (i.e. GF)

```
cd ./app

python baseline_SVM.py
```

the 6 Selected Golden Features:

| Category                | Feature                               | Corresponding field in "src_top_20" pickle file |
| ----------------------- | ------------------------------------- | ----------------------------------------------- |
| Warning characteristics | warning type                          | info["warning_category"]                        |
| Warning characteristics | warning pattern                       | info["violation_type"]                          |
| Warning characteristics | warning priority                      | info["priority"]                                |
| Warning combination     | defect likelihood for warning pattern | info["defect_likelihood_for_warning_pattern"]   |
| Warning combination     | warning context for warning type      | info["warning_context_for_warning_type"]        |
| Warning combination     | warning context in file               | info["warning_context_of_file"]                 |



