# %% [markdown]
# # Model Ensemble Inference Demo
# Minimal example of model inference from a model ensemble using Cerberus.

# %%

from pprint import pprint
from cerberus import model_ensemble
from cerberus.model_ensemble import ModelEnsemble
from cerberus.interval import Interval
from cerberus.dataset import CerberusDataset

# %%
## 1. Load Model Ensemble
model_ensemble = ModelEnsemble(
    checkpoint_path="tests/data/models/chip_ar_mdapca2b_bpnet/multi-fold"
)
# pprint(model_ensemble.folds)
# pprint(model_ensemble.keys())
pprint(model_ensemble.cerberus_config)

# %%
# ## 2. Prepare Dataset
dataset = CerberusDataset(
    genome_config=model_ensemble.cerberus_config["genome_config"],
    data_config=model_ensemble.cerberus_config["data_config"]
)

# %%
input_len = dataset.data_config["input_len"]
iv = Interval(chrom="chr1", start=1000000, end=1000000 + input_len)
print(f"Predicting on interval: {iv}")
res = model_ensemble.predict_intervals([iv], dataset=dataset)
pprint(res)
# %%
