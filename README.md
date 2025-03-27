# [NTIRE 2025 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## How to test the model?

1. `git clone https://github.com/aagamjn/NTIRE-2025`
2. `pip install -r [path to requirements.txt]`
3.  download pretrained weights as told in the below section , ensure it is in proper directory NTIRE-2025/model_zoo/team22_HAT.pth
4.  ensure that you are in correct directory /NTIRE-2025 before running
   ```
    import os
    # Change working directory to NTIRE-2025
    os.chdir("./NTIRE-2025")
    # Verify current directory
    print("Current Directory:", os.getcwd())
   ```
    
5. Run this code to test the model:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python [ path to test.py] --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
   
## Model Pretrained Weights
     # you can use our model pretrain weights from this:
 ```bash
!pip install gdown  # Ensure gdown is installed


import gdown

# Google Drive file ID (extracted from the link)
file_id = "1qtezABZb6xPB99S2qx9mU7tqP9C7CqJZ"
output_name = "team22_HAT.pth"  # Change to desired filename

# Download the file
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_name, quiet=False)
```
## Model Result On DIV2K
    # you can check the model output on DIV2K test dataset :
    `!pip install gdown`
    `!gdown --id 1KJ6_HhnLZuImIANYFJDiiYoyH4AoqANr`

    
## How to eval images using IQA metrics?

### Environments

```sh
conda create -n NTIRE-SR python=3.8
conda activate NTIRE-SR
pip install -r requirements.txt
```


### Folder Structure
```
test_dir
├── HR
│   ├── 0901.png
│   ├── 0902.png
│   ├── ...
├── LQ
│   ├── 0901x4.png
│   ├── 0902x4.png
│   ├── ...
    
output_dir
├── 0901x4.png
├── 0902x4.png
├──...

```

### Command to calculate metrics

```sh
python eval.py \
--output_folder "/path/to/your/output_dir" \
--target_folder "/path/to/test_dir/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
```

The `eval.py` file accepts the following 4 parameters:
- `output_folder`: Path where the restored images are saved.
- `target_folder`: Path to the HR images in the `test` dataset. This is used to calculate FR-IQA metrics.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.

### Weighted score for Perception Quality Track

We use the following equation to calculate the final weight score: 

$$
\text{Score} = \left(1 - \text{LPIPS}\right) + \left(1 - \text{DISTS}\right) + \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right). 
$$

The score is calculated on the averaged IQA scores. 

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
