
# ACE-HOLD
 * This folder contains the original HOLD and the files we modified to optimize the mano registration step.
## How to use our code
 * You can use our code following the steps below:
 * First, clone the referenced HOLD repo:
 ```bash
 git clone https://github.com/zc-alexfan/hold.git
```
* Then, set up the environment according to the original HOLD repo.
* After that, replace registration.py in the original hold folder with our corresponding file, and place `ace_register_mano.py` in the `generator/scripts/` folder alongside the original register_mano.py
* Finally, when following the preprocessing step in `custom_arctic.md`, replace `register_mano.py` in the mano registration step
```bash
pyhold scripts/register_mano.py --seq_name $seq_name --save_mesh --use_beta_loss
```
with our modified script:
```bash
pyhold scripts/ace_register_mano.py --seq_name $seq_name --save_mesh --use_beta_loss
```

## Other small changes
* When preprocessing, you may run into an assertion error while running `validate_masks.py`, when this
happens, you can check the script to see if the image type changed from jpg to png in line 14, for in older versions of the original code the image type is jpg.

