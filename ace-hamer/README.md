
# ACE-HaMeR
 * This folder contains the original HaMeR and the files we modified/implemented to add the relation-aware tokenization module into HaMeR
## How to use our code
 * You can use our code following the steps below:
 * First, clone the referenced HaMeR repo:
 ```bash
 git clone https://github.com/geopavlakos/hamer.git
```
* Then, set up the environment and download HaMeR training data according to the original HaMeR repo.
* After that, replace `hamer.py`, `image_dataset.py` and `vitdet_dataset.py` in the original folder with our corresponding files, and add  `ace_hamer_vit_transformer.yaml` under the `hamer/configs_hydra/experiment/` folder, add `rat.py` and `sir.py` under the `hamer/models/components/` folder. Also, as the tokens are already rearranged by our self-attention encoder, there's no need to rearrange x in `mano_head.py`. Therefore, you need to annotate line 61 in `hamer/models/heads/mano_head.py`.
* Finally, you can start training our modified HaMeR using:
```bash
python train.py exp_name=hamer-rat data=mix_all experiment=ace_hamer_vit_transformer trainer=gpu launcher=local
```
* For training data, you can adjust the weights of different datasets freely. If you want to use the same data we used, you can refer to `mix_all.yaml` in this folder.
