# Did I See It Before? Detecting Previously-Checked Claims over Twitter

This repository reproduces the attained results of the paper entitled "[Did I See It Before? Detecting Previously-Checked Claims over Twitter](https://link.springer.com/chapter/10.1007/978-3-030-99736-6_25)", which is published in proceedings of the 44th European Conference on Information Retrieval (ECIR 2022).

## Requirements

To install the required libraries, create a new Conda environment and give it a meaningful name (I will refer to it by "WM-ECIR22"). Then, you can install the needed libraries within the created environment.

```
# 1. install conda environment 
conda create -n WM-ECIR22 anaconda

# 2. activate the environment
conda activate WM-ECIR22

# 3. install all needed libraries
pip install -r requirements.txt

```

Generally, all paths for input and output files need to be configured in **configure.py**  file.

Despite using the same code, you might not get the exact results reported in the paper due to randomization initialization for neural-based models.

## Datasets

In our experiments, we used two datasets, namely [CheckThat! 2020](https://github.com/sshaar/clef2020-factchecking-task2) and [CheckThat! 2021](https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/tree/master/task2). Some files of these two repositories were copied here to ease running the experiments. So, please consider citing their work.

## 1. Initial Retrieval with Preprocessing

To expand the tweet with some information out of URLs, you need to do the following:

1. Create a Twitter developer account from [here](https://developer.twitter.com/en/apply-for-access).
2. Start the MRIZA (Meta Reverse Image Search API) server using [this repository](https://github.com/vivithemage/mrisa).
3. Specify the input and output query file paths and your Twitter account keys within **tweet_url_extractor.ipynb** notebook.
4. Run all cells of **tweet_url_extractor.ipynb** notebook.

To check the performance of the initial retrieval stage on CT2020-En-dev before and after applying the preprocessing (replicate the results of table 1 of the paper), you need to:

1. Go to the **en_classic_retrieval.ipynb.**
2. Configure the input and output files.
3. Run all cells of the notebook. Please note that the pyterrier indexer

Please note that the pyterrier indexer expects all verified claims to be within one file. You can use the **helper/prepare_data.py** script for that purpose.

## 2. MonoBERT Experiments on English Data

After expanding the tweets in the previous stage, you can proceed and tune BERT as monoBERT reranker. To check the effect of varying the initial retrieval depth on CheckThat! 2020 dataset, you need to:

1. Configure the input and output files in **en_2020_mono_bert.ipynb** notebook.
2. Run all cells to tune multiple BERT variants as monoBERT rerankers. By running this notebook, you will reproduce the results reported in Tables 4 and 5.

Similarly, **en_2021_mono_bert.ipynb** tunes multiple BERT variants as monoBERT rerankers but using CheckThat! 2021 dataset. This notebook reproduces Table 6.

## 3. MonoBERT Experiments on Arabic Data

We examine the effectiveness of the proposed pipeline by applying the attained conclusions from English experiments on the CheckThat! 2021 Arabic dataset. To reproduce the results reported in Table 8, you need to:

1. Configure the input and output files in **ar_2021_mono_bert.ipynb** notebook.
2. Run all cells in that notebook. Here, you will notice the tunning of Arabic BERT models as rerankers.

## Citation

If you used any piece of this repository, please consider citing our work :

```
@inproceedings{mansour2022did,
  title={Did I See It Before? Detecting Previously-Checked Claims over Twitter},
  author={Mansour, Watheq and Elsayed, Tamer and Al-Ali, Abdulaziz},
  booktitle={European Conference on Information Retrieval},
  pages={367--381},
  year={2022},
  organization={Springer},
  isbn="978-3-030-99736-6"
}
```