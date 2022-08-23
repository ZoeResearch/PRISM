# CLR4StaticBug

## Dependency

Tested on ubuntu 18.04

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



### Evaluating embedding model for bug detection

```
cd CLR4StaticBug/

mkdir ./all_embedding/pickle_object

mv YOURPATH/dataset/dataset1 ./all_embedding/pickle_object/spotbugs

mv YOURPATH/pre_trained_model/model4data1 ./pre_train_def/spotbugs

cd ./all_embedding/Word2vec(fasttext/Glove/Elmo/Bert)/src_code

python src_word2vec.py(src_fasttext.py/src_glove.py/src_elmo.py/src_bert.py) --cls spot_bin --code src --fold 11 --retrain False --split_test True --neural lstm (--bert_model bert_seq)

cd spot_bin(spot_mul)/src_w2v_lstm (to see results, results of 10 times are saved in best_score_record)
```

--cls  kind of data, "spot_bin" denotes the binary bug classification, "spot_mul" denotes the multi-class bug classification

--code  kind of code

--fold  the fold number of cross validation

--retrain whether to retrain the pretrained embeddding model

--split_test  whether to split the test set

--neural  specify detecting neural network (lstm/blstm/gru/bgru/textcnn/mlp)

--bert_model  specify the name of pretrained bert model (when specify this parameter, "--neural" needs to be set to "lr")



### Evaluating embedding model for differentiating actionable and unactionable warnings

```
mv YOURPATH/dataset/dataset2 ./app/data

mkdir ./app/model & mv YOURPATH/pre_trained_model/model4data2 ./app/model/src_top_20

cd ./app

python sort_bug.py --filter 0 --balance 1 --train 1 ---method embedding --code src --embed w2v --detect gru 

cd ./score/w2v_gru/(to see results, results of 10 times are saved in best_score_record)
```

--filter:  whether to filter the cross-method bug

--balance:  whether to balance the dataset

--train:  whether to train detecting model 

--method:  run our embedding model or baseline model

--code:  kind of code

--embed:  kind of embedding model (w2v/fasttext/glove/elmo/bert_seq)

--detect:  kind of detecting model (lstm/blstm/gru/bgru/textcnn)



## Evaluating PRISM and baselines

- running PRISM

```
cd ./all_embedding/Hybrid/new/

python voting.py --vote True --embed group3

cd ../../app/vote_score/vote_group3/ (to see results, results of 10 times are saved in hard_vote_results and soft_vote_results)
```

--vote: whether to vote

--embed: the kind of setting you choose, "w2v_bgru" denotes the best CRL model,  "group3", "group5", "group7", "group9" denote "M" parameter of PRISM (i.e., to combine the top-3,top-5, top-7, top-9 CRL models together to make voting and classification)



- running baseline1: HWP 	

```
cd ./app_embedding/app/

python sort_bug.py --filter 0 --balance 1 --train 0 ---method hwp --code src 
```

- running baseline2: Golden features

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



