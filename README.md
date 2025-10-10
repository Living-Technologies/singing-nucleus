# singing-nucleus
This repository is a collection of simple ai models that do specific tasks. 

## How do I use this code

This is not user friendly and it targets very specific data. For a general outline.

- create a set of image/mask crops
- create a "quick mesh" version of the masks.
- Train an img_2_dt model based on the input image and the masks
- Train a bin_2_mesh model to produce meshes from the binary masks
- Train an img_2_mesh model based off of the two config files above.