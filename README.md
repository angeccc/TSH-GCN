This is the code for the paper [Statistically guided lightweight skeleton graph learning for monocular 3D human pose estimation](a submission to The Visual Computer) in Pytorch.
### Dependencies

* cuda 9.0
* Python 3.6
* [Pytorch](https://github.com/pytorch/pytorch) 0.4.1.

### Dataset setup
Datasets are provided by [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) by Pavllo etal., which can be downloaded by:
```bash
cd data
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
cd ..
```
3D labels and ground truth can be downloaded and put in dataset/ folder [3d gt labels](https://drive.google.com/file/d/1P7W3ldx2lxaYJJYcf3RG4Y9PsD4EJ6b0/view?usp=sharing)

### Test the Model
To test the trained network, run:
```bash
python main_graph.py --pad 0 --show_protocol2  --reload_trained_stgcn 1  --trained_model_dir 'results/1_frame/two_scale_gcn/no_pose_refine/cpn/' --stgcn_model 'model_st_gcn_40_eva_post_5065.pth' 
```

### Train the Model
To train the TSH-GCN, run:
```bash
python main_graph.py --pad 0 --pro_train 1 --save_trained_model 1
```

### Citation
```bash
{
  title={Statistically guided lightweight skeleton graph learning for monocular 3D human pose estimation},
  author={Ange Chen, Chengdong Wu, Chuanjiang Leng and Xiaosheng Yu}
}
```
### Acknowledgements
Some of our implementation code/preprocessed data was adapted from https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks by Cai et al. Thanks for their help!

### Licence
MIT
