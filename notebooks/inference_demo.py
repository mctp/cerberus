# %% [markdown]
# # Model Ensemble Inference Demo
# Minimal example of model inference from a model ensemble using Cerberus.

# %%

from pprint import pprint
from cerberus import model_ensemble
from cerberus.model_ensemble import ModelEnsemble
from cerberus.interval import Interval
from cerberus.dataset import CerberusDataset
try:
    from paths import get_project_root
except ImportError:
    from notebooks.paths import get_project_root

# %%
## 1. Load Model Ensemble
project_root = get_project_root()
checkpoint_path = project_root / "tests/data/models/chip_ar_mdapca2b_bpnet/multi-fold"
if not checkpoint_path.exists():
    print(f"Skipping: checkpoint directory not found: {checkpoint_path}")
    print("Run 'bash examples/chip_ar_mdapca2b_bpnet.sh' with a multi-fold setup to generate it.")
    import sys; sys.exit(0)
model_ensemble = ModelEnsemble(
    checkpoint_path=checkpoint_path,
    search_paths=[project_root],
)
# pprint(model_ensemble.folds)
# pprint(model_ensemble.keys())
pprint(model_ensemble.cerberus_config)

# %%
# ## 2. Prepare Dataset
dataset = CerberusDataset(
    genome_config=model_ensemble.cerberus_config.genome_config,
    data_config=model_ensemble.cerberus_config.data_config,
)

# %%
input_len = dataset.data_config.input_len
iv = Interval(chrom="chr1", start=1000000, end=1000000 + input_len)
print(f"Predicting on interval: {iv}")
res = model_ensemble.predict_intervals([iv], dataset=dataset)
pprint(res)
# %%
