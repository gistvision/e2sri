# Events to Super-Resolved Images (E2SRI)
This is a code repo for **[Learning to Super Resolve Intensity Images from Events](http://openaccess.thecvf.com/content_CVPR_2020/papers/I._Learning_to_Super_Resolve_Intensity_Images_From_Events_CVPR_2020_paper.pdf)** ([CVPR 2020 - Oral](https://youtu.be/kiSCXegcwfM))<br>
[Mohammad Mostafavi](https://smmmmi.github.io/), [Jonghyun Choi](http://ppolon.github.io/) and [Kuk-Jin Yoon](http://vi.kaist.ac.kr/project/kuk-jin-yoon/) (Corresponding author)

[![E2SRI](https://github.com/gistvision/e2sri/blob/master/images/E2SRI.png)](https://youtu.be/ZMFAseI1DM8)
 
Our extended and upgraded version produces highly consistent videos, and includes further details and experiments [E2SRI: Learning to Super-Resolve Intensity Images from Events - TPAMI 2021](https://www.computer.org/csdl/journal/tp/5555/01/09485034/1veokqDc14Q)


If you use any of this code, please cite both following publications:

```@article{mostafaviisfahani2021e2sri,
  title={E2SRI: Learning to Super-Resolve Intensity Images from Events},
  author={Mostafaviisfahani, Sayed Mohammad and Nam, Yeongwoo and Choi, Jonghyun and Yoon, Kuk-Jin},
  journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence},
  number={01},
  pages={1--1},
  year={2021},
  publisher={IEEE Computer Society}
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
```

## Inference
- Download the linked material below
  * Pretrained weight ([2x_7S_weight.zip](https://drive.google.com/file/d/1rlPQoQtw496AWrRor3jMFfF2ETgpLBMx/view?usp=sharing)) for 2x scale (2x width and 2x height) and 7S sequences of stacks.
  * A sample real-world sequence of stacks ([slider_depth.zip](https://drive.google.com/file/d/1YLXeY7bK4QyN26l9ILHD-tmc4Suwdch-/view?usp=sharing)).

- Unzip and put the files in 2x_7S_weight.zip (Final.pth) to the weight folder (/e2sri/weight/Final.pth)

- Unzip the sample event stack folder slider_depth and put it in the root folder (/e2sri/slider_depth)

- Run reconstruction:

```bash
python test.py --data_dir ../slider_depth --checkpoint_dir .. --save_dir ../output
```

Note that our code with the given weights (7S) consumes ~ 4753MiB GPU memory at inference.

From this sample event stack, you should produce a similar (resized) result as:

<img src="https://github.com/gistvision/e2sri/blob/master/images/event.png"> <img src="https://github.com/gistvision/e2sri/blob/master/images/sample.png" width="240" height="180">


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

- [Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds - Openaccess ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)

- [Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds - Openaccess ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)

- [Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds - Openaccess ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)

- [Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds - Openaccess ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)

- [Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds - Openaccess ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)



## License


MIT license.


