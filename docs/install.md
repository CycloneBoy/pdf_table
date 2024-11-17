# install

## install
```shell

conda create -n torch python=3.11
conda activate torch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

```shell

OSError: Ghostscript is not installed. You can install it using the instructions here: https://camelot-py.readthedocs.io/en/master/user/install-deps.html

apt install ghostscript

```

```shell
pdftable --output_dir /nlp_data/pdftable/outputs/pdf/inference_results/pdf_debug/2024-11-17/20241117_143700 \
--lang ch \
--file_path_or_url https://disc.static.szse.cn/disc/disk03/finalpage/2023-06-09/52a65d41-637d-4e66-879b-5a5744092d2e.PDF 


pdftable --output_dir /nlp_data/pdftable/outputs/pdf/inference_results/pdf_debug/2024-11-17/20241117_200100 \
--lang en \
--recognizer_model ConvNextViT \
--table_structure_model SLANet \
--file_path_or_url https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf 

```