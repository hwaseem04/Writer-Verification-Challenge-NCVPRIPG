# Writer-Verification-Challenge-NCVPRIPG

## Model information
- It is a two stage model. [CRAFT](https://github.com/clovaai/CRAFT-pytorch) is used to extract words from each images. Each word of writers are then paired appropriately (similar / dissimilar). 
- Each pair is tokenised, appended as a single token and passed into a `Vision transformer`. 
- Average pooling is finally applied on respective token outputs from encoder from which euclidean distance is calculated. Contrastive loss with a margin is used for training.

## Requirements
- Python 3.9
- [requirements.txt]()
- [PyTorch CRAFT](https://github.com/clovaai/CRAFT-pytorch)

## Setup
- Conda Environment setup, requiremnets installation, CRAFT setup
```bash
conda create -name writer python=3.9
conda activate writer
```
```python
pip install -r requirements.txt
```
- Download weights for craft from [here](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ). Place it at `craft_model/` directory.

- Download weights for transformer model from [here](https://drive.google.com/file/d/1hhP92cN1I_KxRkyq8ChttRL3NOvtEa-v/view?usp=sharing). Place it in root directory itself.

## Important file information
- `config.py` is very useful if you are running your code in GPU cluster where you can `sbatch` your main.py with a script file (Do not need to think about it if you dont have this setup, following below step is more than enough).
- But make sure to have necessary information in config.py file since most of the code files access command line arguments (it is mostly self-explanatory). 

## For training
- Obtain results from craft model and prepare dataset;
    - example command `python process_data.py` 
- example command `python main.py --data_path "dataset/"  --batch_size 2`

## For testing/evaluation
- Need to first generate craft results and store it for quite faster inference
    - example command `python prepare_test_data.py --test_path "dataset/semi_test" --csv_path "dataset/test.csv" --data_path "dataset/"`
- Now run `python predict.py --csv_path "dataset/test.csv" --test_path "dataset/semi_test" --batch_size 1 --data_path "dataset/"`
- Results are stored in csv file.