# LLMDTA

LLMDTA: Improving Cold-Start Prediction in Drug-Target Affinity with Biological LLM.

![](./LLMDTA-model.png)

## Requirements

gensim==4.3.1 \
matplotlib==3.2.2 \
mol2vec==0.1 \
numpy==1.23.4 \
pandas==1.5.2 \
rdkit==2023.3.2 \
scikit_learn==1.2.2 \
scipy==1.8.1 \
torch==1.8.2 \
tqdm==4.65.0 \

Install [ESM2](https://github.com/facebookresearch/esm) from repo.

## Example Usage

### Training with Davis\KIBA\Metz

1. Download Dataset from this [site](https://www.kaggle.com/datasets/christang0002/llmdta/data).

2. Config `hyperparameter.py`

```
self.data_root       : the dataset pth
self.dataset         : the dataset name, davis, kiba or metz
self.running_set     : the task setting, warm, novel-drug, novel-prot or novel-pair

self.mol2vec_dir    : the extracted moleculars pretraining feature
self.protvec_dir    : the targets pretraining feature
self.drugs_dir      : drugs list
self.prots_dir      : targets list

self.cuda           : set the gpu device
```

3. RUN `python train.py`

### Training from Your Dataset

If you want to use LLMDTA to training on your dataset. You should prepare the dataset and pretrained feature at first.

In code_prepareEmb folder, we provide notbooks to generate pretrained embedding and split 5 fold cold-start dataset.

1. Prepare Customized Dataset.\
   We expect the training data should using the following format:

- `drugs.csv`\
  ['drug_id', 'drug_smile']
- `targets.csv`\
  ['prot_id','prot_seq']
- `pairs.csv`\
  ['drug_id', 'prot_id', 'label']

2. Generate Pretraining Embeddings. \
   Using `code_prepareEmb/_PreparePretrain.ipynb` to extract the mol2vec and ESM2 pretraining features from drugs.csv and targets.csv.

3. Config `hyperparameter.py`

4. RUN. `python train.py`

### Prediction

1. Prepare the drugs and targets in CSV like format.
2. Config the `hyperparameter4pred.py` file.

```
self.word2vec_pth   : the pretrained word2vec feature
self.pred_dataset   : the prediction task name, for saving result
self.sep            : the separator while reading drugs/target list

self.pred_drug_dir  : the drugs list file
self.pred_prot_dir  : the target list file
self.d_col_name     : the col names of drugs file
self.p_col_name     : the col names of targets file

self.model_fromTrain    : the pretrained model
```

3. Run Prediction. And the result file will saving in current path.

```
python code/pred.py
```
