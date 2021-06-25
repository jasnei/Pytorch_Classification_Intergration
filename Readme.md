# Some Classification Intergration

## Folders & Files description

- Folders
    - checkpoints
        - here to save training checkpoint, log, etc
    - core
        - training, validating fuctions are in this folder
    - loss
        - some loss fuctions
    - models
        - all models are in this folder, you could import from models
    - my_data
        - train data, test data folder
    - result
        - predict result
    
- Files
    - mean_std_calculation.ipynb
        - calculate train set mean & std
    - predict_classification.py
        - predict are processing in this script
    - test.py
        - test script, you could run some test in this scrip, if you want to
    - torch_convert.py
        - get all the training images and labels, and split train test set
    - torch_dataset.py
        - pytorch dataset setupt according tor data format
    - train_efficient.py
        - train script, here is efficientnet, so name train_efficient
    - train_efficientnet_album.py
        - train script, efficientnet training using albumentations for data augmentation