# External model definitions

Several scripts import baseline and shared model definitions that are not yet
in this repository. Place these files in this folder (or point
`MAP_EXTERNAL_DIR` at the folder that contains them):

| File | Provides | Used by |
|---|---|---|
| `dncnn_model.py` | `DnCNN` | baseline training, cross-dataset eval, tables/figures |
| `restormer_lite_model.py` | `RestormerLite` | baseline + range training, eval, tables/figures |
| `swinir_lite_model.py` | `SwinIRLite` | baseline + range training, eval, tables |
| `unet_model.py` | `UNet2D`, `dice_coefficient` (segmentation) | downstream Dice evaluation |

The MAP-DenoiseGAN generator, discriminator, noise model, noise estimator and
MoE analysis scripts do **not** depend on these files.
