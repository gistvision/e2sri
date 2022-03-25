# Events to Super-Resolved Images (E2SRI)
This is a code repo for **[Learning to Super Resolve Intensity Images from Events](http://openaccess.thecvf.com/content_CVPR_2020/papers/I._Learning_to_Super_Resolve_Intensity_Images_From_Events_CVPR_2020_paper.pdf)** ([CVPR 2020 - Oral](https://youtu.be/kiSCXegcwfM))<br>
[Mohammad Mostafavi](https://smmmmi.github.io/), [Jonghyun Choi](http://ppolon.github.io/) and [Kuk-Jin Yoon](http://vi.kaist.ac.kr/project/kuk-jin-yoon/) (Corresponding author)

[![E2SRI](https://github.com/gistvision/e2sri/blob/master/images/E2SRI.png)](https://youtu.be/ZMFAseI1DM8)
 
Our extended and upgraded version produces highly consistent videos, and includes further details and experiments [E2SRI: Learning to Super-Resolve Intensity Images from Events - TPAMI 2021](https://www.computer.org/csdl/journal/tp/5555/01/09485034/1veokqDc14Q)


If you use any of this code, please cite both following publications:

```bibtex
@article{mostafaviisfahani2021e2sri,
  title     = {E2SRI: Learning to Super-Resolve Intensity Images from Events},
  author    = {Mostafaviisfahani, Sayed Mohammad and Nam, Yeongwoo and Choi, Jonghyun and Yoon, Kuk-Jin},
  journal   = {IEEE Transactions on Pattern Analysis \& Machine Intelligence},
  number    = {01},
  pages     = {1--1},
  year      = {2021},
  publisher = {IEEE Computer Society}
}
```

```bibtex
@article{mostafavi2020e2sri,
  author  = {Mostafavi I., S. Mohammad and Choi, Jonghyun and Yoon, Kuk-Jin},
  title   = {Learning to Super Resolve Intensity Images from Events},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  month   = {June},
  year    = {2020},
  pages   = {2768-2786}
}
```


### Maintainer
* [Mohammad Mostafavi](https://smmmmi.github.io/)
* [Yeong-oo Nam](https://gistvision.github.io/people.html)

## Set-up

- Make your own environment

```bash
python -m venv ./e2sri
source e2sri/bin/activate
```

- Install the requirements
```bash
cd e2sri
pip install -r requirements.txt
```

- Unizp pyflow
```bash
cd src
unzip pyflow.zip
cd pyflow
python3 setup.py build_ext -i
```

## Preliminary
- Download the linked material below
  * Sample pretrained weight ([2x_7s.pth](https://drive.google.com/file/d/1fCPGoAynMVLI_23vuDsceL_39rkt1QJl/view?usp=sharing)) for 2x scale (2x width and 2x height) and 7S sequences of stacks.
  * Sample dataset for training and testing ([datasets.zip](https://drive.google.com/file/d/1d11Ec-vUHPNmIDJZi-YSQ_50WdOxHdlo/view?usp=sharing)).

- Unzip the dataset.zip file and put the pth weight file in the main folder

```bash
unzip dataset.zip
cd src
```

## Inference
- Run inference:
```bash
python test.py --data_dir ../dataset/slider_depth --checkpoint_path ../save_dir/2x_7s.pth --save_dir ../save_dir
```

Note that our code with the given weights (7S) consumes ~ 4753MiB GPU memory at inference.

From this sample event stack, you should produce a similar (resized) result as:

<img src="https://github.com/gistvision/e2sri/blob/master/images/event.png"> <img src="https://github.com/gistvision/e2sri/blob/master/images/sample.png" width="240" height="180">

## Training
- Run training:
```bash 
python3 train.py --config_path ./configs/2x_3.yaml --data_dir ../dataset/Gray_5K_7s_tiny --save_dir ../save_dir
```


## Event Stacking

We provided a sample sequence ([slider_depth.zip](https://drive.google.com/file/d/1YLXeY7bK4QyN26l9ILHD-tmc4Suwdch-/view?usp=sharing)) made from the rosbags of the [Event Camera Dataset and Simulator](http://rpg.ifi.uzh.ch/davis_data.html). The rosbag (bag) is a file format in ROS (Robot Operating System) for storing ROS message data.
You can make other sequences using the given matlab m-file (/e2sri/stacking/make_stacks.m).
The matlab code depends on [matlab_rosbag](https://github.com/bcharrow/matlab_rosbag/releases) which is included in the stacking folder and needs to be unzipped.

**Note:**
The output image quality relies on "events_per_stack" and "stack_shift". We used "events_per_stack"=5000, however we did not rely on "stack_shift" as we synchronized with APS frames instead. The APS synchronized stacking when this 5000 setting should be kept will be released with the training code together.


## Datasets

A list of publicly available event datasets for testing:

- [Bardow et al., CVPR'16](http://wp.doc.ic.ac.uk/pb2114/datasets/)
- [The Event Camera Dataset and Simulator](http://rpg.ifi.uzh.ch/davis_data.html)
- [Multi Vehicle Stereo Event Camera Dataset, RAL'18](https://daniilidis-group.github.io/mvsec/download/)
- [Scherlinck et al., ACCV'18](https://drive.google.com/drive/folders/1Jv73p1-Hi56HXyal4SHQbzs2zywISOvc)
- [High Speed and HDR Dataset](http://rpg.ifi.uzh.ch/E2VID.html)
- [Color event sequences from the CED dataset Scheerlinck et al., CVPR'18](http://rpg.ifi.uzh.ch/data/E2VID/datasets/CED_CVPRW19/)


## Related publications

- [Stereo Depth from Events Cameras: Concentrate and Focus on the Future]() + [Code]() - CVPR 2022 (TBU)

- [Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds - Openaccess ICCV 2021 (PDF)](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)

- [E2SRI: Learning to Super Resolve Intensity Images from Events - TPAMI 2021 (Link)](https://www.computer.org/csdl/journal/tp/5555/01/09485034/1veokqDc14Q)

- [Learning to Reconstruct HDR Images from Events, with Applications to Depth and Flow Prediction - IJCV 2021](http://vi.kaist.ac.kr/wp-content/uploads/2021/04/Mostafavi2021_Article_LearningToReconstructHDRImages-1.pdf)

- [Learning to Super Resolve Intensity Images from Events - Openaccess CVPR 2020 (PDF)](https://openaccess.thecvf.com/content_CVPR_2020/papers/I._Learning_to_Super_Resolve_Intensity_Images_From_Events_CVPR_2020_paper.pdf)

- [Event-Based High Dynamic Range Image and Very High Frame Rate Video Generation Using Conditional Generative Adversarial Networks - Openaccess CVPR 2019 (PDF)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Event-Based_High_Dynamic_Range_Image_and_Very_High_Frame_Rate_CVPR_2019_paper.pdf)


## License


MIT license.
