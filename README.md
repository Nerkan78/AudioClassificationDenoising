# AudioClassificationDenoising

Module `multiplication.py` contains function `multiplicate` which is the solution to the first problem. Since in the problem was not specified what is to return when the length of `A` is one, the `[1]` is returned.

Modules `train.py` and `eval.py` implement the solution to the second problem. Running `train.py` will train and save model for the corresponding task. Running `eval.py` will evaluate and report the perfomance. Note that for `eval.py` you should provide path for the pretrained model.

Example of run
```
python train.py --train_folder PATH_TO_TRAIN_FOLDER --val_folder PATH_TO_VAL_FOLDER --mode MODE --model_name NAME
python eval.py --test_folder PATH_TO_TEST_FOLDER --mode MODE --model_name NAME
```
where `MODE` is either `classification` or `denoising` and `NAME` is the path to model file without extension. 

