# rob-margin

This repository contains code for "[Understanding Adversarial Robustness: The Trade-off between Minimum and Average Margin][paper]".

[paper]: https://drive.google.com/file/d/1LqPyPmWssBSIhmEeB8e7EcAW6YHPq2kS/view

To run the estimation:
```
bash estimate.sh
```
The script will estimate the average lower bound of the classifier in `./Model/MNISTLR/MNISTLR_01000.tar` and output the result in `./Output` folder.

For training and testing models, see `train.sh` and `test.sh`.
