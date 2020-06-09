# Tex2Shape

This repository contains code corresponding to the paper [**Tex2Shape: Detailed Full Human Body Geometry from a Single Image**](https://arxiv.org/abs/1904.08645).

## Installation

Download and unpack the SMPL model from http://smpl.is.tue.mpg.de/ and link the files to the `vendor` directory.
```
cd vendor/smpl
ln -s <path_to_smpl>/smpl_webuser/*.py .
```

Download the neutral SMPL model from http://smplify.is.tue.mpg.de/ and place it in the `assets` folder.
```
cp <path_to_smplify>/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl assets/neutral_smpl.pkl
```

Download pre-trained model weights from [here](https://drive.google.com/open?id=1yl4m7rzr-F9qbBqH-NzRqUQiD5uTTW8P) and place them in the `weights` folder.

```
unzip <downloads_folder>/weights_tex2shape.zip -d weights
```

## Requirements

* Python 2.7
* tensorflow
* keras
* chumpy
* openCV

### Installation through conda

```
conda create --name py2 python=2.7
conda activate py2
pip install tensorflow-gpu
pip install keras
pip install chumpy
pip install opencv-python
```


## Usage

We provide a run script (`run.py`) and sample data for single subject and batch processing.
The script outputs usage information when executed without parameters.

### Quick start

We provide sample scripts for both modes:

```
bash run_demo.sh
bash run_batch_demo.sh
```

## Data preparation

If you want to process your own data, some pre-processing steps are needed:

1. Crop your images to 1024x1024px.
2. Run [DensePose](http://densepose.org/) on your images.

Cropped images and DensePose IUV detections form the input to Tex2Shape. See `data` folder for sample data.

### Image requirements

The person in the image should be roughly facing the camera, should be fully visible, and cover about 70-80% of the image height.
Avoid heavy lens-distortion, small focal-legths, or uncommon viewing angles for better performance.
If multiple people are visible, make sure the IUV detections only contain the person of interest.

## Citation

This repository contains code corresponding to:

T. Alldieck, G. Pons-Moll, C. Theobalt, and M. Magnor. [**Tex2Shape: Detailed Full Human Body Geometry from a Single Image**](https://arxiv.org/abs/1904.08645). In *IEEE International Conference on Computer Vision*, 2019.

Please cite as:

```
@inproceedings{alldieck2019tex2shape,
  title = {Tex2Shape: Detailed Full Human Body Geometry from a Single Image},
  author = {Alldieck, Thiemo and Pons-Moll, Gerard and Theobalt, Christian and Magnor, Marcus},
  booktitle = {{IEEE} International Conference on Computer Vision ({ICCV})},
  year = {2019}
}
```


## License

Copyright (c) 2019 Thiemo Alldieck, Technische Universität Braunschweig, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **Tex2Shape: Detailed Full Human Body Geometry from a Single Image** paper in documents and papers that report on research using this Software.
