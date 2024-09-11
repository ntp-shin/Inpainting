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
- Recommended Python version 3.7 with conda environment
- PyQt5 version 5.15.7
- opencv-python version 4.1.2.30
- torch version 1.7.1+cu110
- torchvision version 0.8.2+cu110
- qimage2ndarray version 1.8.3

### Implementation
The frontend GUI is developed using PyQt. The backend inpainting operations are done using *OpenCV* library. Currently, *OpenCV* provides two algorithms for inpainting which are-
* cv2.INPAINT_TELEA: An image inpainting technique based on the fast marching method (Telea, 2004)
* cv2.INPAINT_NS: Navier-stokes, Fluid dynamics, and image and video inpainting (Bertalmío et al., 2001)

I've mentioned how you can quickly incorporate other inpainting algorithms with this GUI applications down below. Later on, I'll try to incorporate recent deep learning methods that perform way better than these classical image processing algorithms. 

<h4 align="center">
<p>let's see an exmaple
</h4>
<p align="center">
 <img alt="editing" src="https://github.com/Zedd1558/Image-Inpainter/blob/master/demo/editpage.jpg">
</p>
<h4 align="center">
<p>removes text quite well!
</h4>

### Required libraries
PyQt, Numpy, OpenCV3, qimage2ndarray

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


### Contribute
Feel free to fork the project and contribute. You can incorporate recent deep learning methods, make the GUI more easy to use, include relevant photo editing features. 

