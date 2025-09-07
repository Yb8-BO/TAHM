# TACI
# Implementation
## Environment

Python >= 3.7

torch == 1.11.0+cu113 (We haven't tested the code on the lower version of torch)

numpy == 1.20.1

gensim = 4.2.0

tqdm == 4.59.0

pandas == 1.2.4

## Datasets

- Processed Beauty, Sports and two different version of Yelp datasets are included in `data` folder. 

- You can use the code in `data_process` folder to process your own dataset, and we explained its role at beginning of each code file.

- For Yelp dataset, we give two processed datasets from two different versions of Yelp. Yelp-A is processed from Yelp2020, which has 316,354 interactions. Yelp-B is processed from Yelp2022, which has 207,045 interactions.
## Train Model

- Change to `src` folder and Run the following command. (The program will read the data file according to [DATA_NAME]. [Model_idx] and [GPU_ID] can be specified according to your needs)
  
  ```
  python main.py --data_name=[DATA_NAME] --model_idx=[Model_idx] --gpu_id=[GPU_ID]
  ```

  ```
  Example:
  python main.py --data_name=Beauty --model_idx=1  --gpu_id=0
  ```

- The code will output the training log, the log of each test, and the `.pt` file of each test. You can change the test frequency in `src/main.py`.
- The meaning and usage of all other parameters have been clearly explained in `src/main.py`. You can change them as needed.

