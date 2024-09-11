<h1 align="center">
<p>Inpainting
</h1>
<h3 align="center">
<p>A Python GUI application to inpaint images. (Using MAT and CS-MAT model)
</h3>

*Inpainting* is a set of image processing algorithms where damaged, missing or unwanted parts of an image are filled in using the neighbouring pixels. It can also be used to 
remove forground objects. This is a GUI application that helps you do just that. It has an interactive and user-friendly GUI for marking the regions of the images. 

<p align="center">
 <img alt="cover" src="https://github.com/Zedd1558/Image-Inpainter/blob/master/demo/cover.jpg" height="50%" width="50%">
</p>

### Requirements
***You must have Nvidia GPU with CUDA support to run the inpainting algorithms.***
- Recommended Python version 3.7 with conda environment
- PyQt5 version 5.15.7
- opencv-python version 4.1.2.30
- torch version 1.7.1+cu110
- torchvision version 0.8.2+cu110
- qimage2ndarray version 1.8.3

<h4 align="center">
<p>let's see an exmaple
</h4>
<p align="center">
 <img alt="editing" src="https://github.com/Zedd1558/Image-Inpainter/blob/master/demo/editpage.jpg">
</p>
<h4 align="center">
<p>removes text quite well!
</h4>

### Prepare the pre-trained models
1. Download the pre-trained models from the following links:
- [MAT model](https://1drv.ms/u/c/faa4073c72266603/EYOjBit6I75CnzWHbDtadRIBWmwzMqGFOR-4_Te8knKSiw?e=2Q24Yz)
- [CS-MAT model](https://1drv.ms/u/c/faa4073c72266603/ETBcdK3KB49JmKbLNGqRWAMBggvA1CJLp8V_C1dsV1TTXw?e=TVm3jQ)

2. Create 'model' folder in the project directory and place the downloaded models in it. Like this:
``` bash
Inpainting
|__ model
    |__ mat-4m92.pkl
    |__ cs-mat-4m2(new-loss).pkl
|__ inpainter.py  
```


### How to run
1. Clone the repository
```
git clone https://github.com/ntp-shin/Inpainting
```
2. Create a conda environment with python 3.7
```
conda create -n inpainting python=3.7
```
3. Activate the environment
```
conda activate inpainting
```
4. Install the required libraries
```
pip install -r requirements.txt
```
5. Run the inpainter.py file. You can open up console in the project directory and enter this 
```
python inpainter.py
```
<p align="center">
 <img alt="editing" src="https://github.com/Zedd1558/Image-Inpainter/blob/master/demo/inpaint_demo2.gif">
</p>

### Fix Bug
**1. If you use Windows OS and have an error like this**: 
``` bash
Building wheels for collected packages: PyQt5-sip
  Building wheel for PyQt5-sip (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Building wheel for PyQt5-sip (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [5 lines of output]
      running bdist_wheel
      running build
      running build_ext
      building 'PyQt5.sip' extension
      error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for PyQt5-sip
Failed to build PyQt5-sip
ERROR: Could not build wheels for PyQt5-sip, which is required to install pyproject.toml-based projects
```

Try to fix it by installing the Microsoft Visual C++ 14.0 or greater:
- You can download it from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Open the installer and select the "Desktop development with C++" workload

### References
- [Inpainter](https://github.com/zahid58/Inpainter) This project is inspired by this repository.
- [MAT](https://arxiv.org/abs/2203.15270) The MAT model is proposed in this paper.
- CS-MAT is our modification of the MAT model. 