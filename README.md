# SIV
 Pretrained models and testing data for SIV
# Environment
    You can choose cudatoolkit version to match your server. The code is tested on PyTorch 1.10.1+cu113.

    conda create -n spike2flow python==3.9
    conda activate spike2flow
    # You can choose the PyTorch version you like, we recommand version >= 1.10.1
    # For example
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    pip3 install -r requirements.txt

# Preparation
    You can download the required test data and pre-trained models from this anonymous link. Please place them in the 'test_data' and 'pretrained' directories, respectively.

# Evaluate
    For example, to evaluate "Problem 1 dt=21", you may run
    python3 test.py --eval --pretrained pretrained/dt\=21/siv_p1_epoch100.pth --configs configs/Problem1.yml --dt 21
