# Vchitect-2.0: Parallel Transformer for Scaling Up Video Diffusion Models

<!-- <p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p> -->

<div>
<div align="center">
    <a href='https://vchitect.intern-ai.org.cn/' target='_blank'>Vchitect Team<sup>1</sup></a>&emsp;
</div>
<div>
<div align="center">
    <sup>1</sup>Shanghai Artificial Intelligence Laboratory&emsp;
</div>
 
 
<div align="center">
                      <a href="https://arxiv.org/abs/2501.08453">Paper</a> | 
                      <a href="https://vchitect.intern-ai.org.cn/">Project Page</a> |
                      <a href="https://huggingface.co/datasets/Vchitect/Vchitect_T2V_DataVerse">Dataset</a>
</div>

---

![](https://img.shields.io/badge/Vchitect2.0-v0.1-darkcyan)
![](https://img.shields.io/github/stars/Vchitect/Vchitect-2.0)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVchitect-2.0&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Generic badge](https://img.shields.io/badge/DEMO-Vchitect2.0_Demo-<COLOR>.svg)](https://huggingface.co/spaces/Vchitect/Vchitect-2.0)
[![Generic badge](https://img.shields.io/badge/Checkpoint-red.svg)](https://huggingface.co/Vchitect/Vchitect-XL-2B)





## 🔥 Update and News
- [2025.03.17] 🔥 Our [Vchitect-T2V-Dataverse](https://huggingface.co/datasets/Vchitect/Vchitect_T2V_DataVerse) is released.
- [2025.01.25] Our [paper](https://arxiv.org/abs/2501.08453) is released.
- [2024.09.14] Inference code and [checkpoint](https://huggingface.co/Vchitect/Vchitect-XL-2B) are released.

## :astonished: Gallery

<table class="center">

<tr>

  <td><img src="assets/samples/sample_0_seed3.gif"> </td>
  <td><img src="assets/samples/sample_1_seed3.gif"> </td>
  <td><img src="assets/samples/sample_3_seed2.gif"> </td> 
</tr>


        
<tr>
  <td><img src="assets/samples/sample_4_seed1.gif"> </td>
  <td><img src="assets/samples/sample_4_seed4.gif"> </td>
  <td><img src="assets/samples/sample_5_seed4.gif"> </td>     
</tr>

<tr>
  <td><img src="assets/samples/sample_6_seed4.gif"> </td>
  <td><img src="assets/samples/sample_8_seed0.gif"> </td>
  <td><img src="assets/samples/sample_8_seed2.gif"> </td>      
</tr>

<tr>
  <td><img src="assets/samples/sample_12_seed1.gif"> </td>
  <td><img src="assets/samples/sample_13_seed3.gif"> </td>
  <td><img src="assets/samples/sample_14.gif"> </td>    
</tr>

</table>


## Installation

### 1. Create a conda environment and install PyTorch

Note: You may want to adjust the CUDA version [according to your driver version](https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version).

  ```bash
  conda create -n VchitectXL -y
  conda activate VchitectXL
  conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
  ```

### 2. Install dependencies

  ```bash
  pip install -r requirements.txt
  ```

## Inference
**First download the [checkpoint](https://huggingface.co/Vchitect/Vchitect-XL-2B).**
~~~bash

save_dir=$1
ckpt_path=$2

python inference.py --test_file assets/test.txt --save_dir "${save_dir}" --ckpt_path "${ckpt_path}"
~~~

In inference.py, arguments for inference:
  - **num_inference_steps**: Denoising steps, default is 100
  - **guidance_scale**: CFG scale to use, default is 7.5
  - **width**: The width of the output video, default is 768
  - **height**: The height of the output video, default is 432
  - **frames**: The number of frames, default is 40

The results below were generated using the example prompt.

<table class="center">

<tr>

  <!-- <td><img src="assets/samples/sample_0_seed2.gif"> </td> -->
  <td><img src="assets/samples/sample_31_seed0.gif"> </td>
  <td><img src="assets/samples/sample_33_seed2.gif"> </td> 
</tr>

<tr>
  <!-- <td>There is a painting depicting a turtle swimming in ocean.</td> -->
  <td>A snowy forest landscape with a dirt road running through it. The road is flanked by trees covered in snow, and the ground is also covered in snow. The sun is shining, creating a bright and serene atmosphere. The road appears to be empty, and there are no people or animals visible in the video. </td>
  <td>The video opens with a breathtaking view of a starry sky and vibrant auroras. The camera pans to reveal a glowing black hole surrounded by swirling, luminescent gas and dust. Below, an enchanted forest of bioluminescent trees glows softly. The scene is a mesmerizing blend of cosmic wonder and magical landscape.</td>      
</tr>
</table>




The base T2V model supports generating videos with resolutions up to 720x480 and 8fps. Then，[VEnhancer](https://github.com/Vchitect/VEnhancer) is used to upscale the resolution to 2K and interpolate the frame rate to 24fps.

## BibTex
```
@article{fan2025vchitect,
  title={Vchitect-2.0: Parallel Transformer for Scaling Up Video Diffusion Models},
  author={Fan, Weichen and Si, Chenyang and Song, Junhao and Yang, Zhenyu and He, Yinan and Zhuo, Long and Huang, Ziqi and Dong, Ziyue and He, Jingwen and Pan, Dongwei and others},
  journal={arXiv preprint arXiv:2501.08453},
  year={2025}
}
```

## 🔑 License

This code is licensed under Apache-2.0. The framework is fully open for academic research and also allows free commercial usage.


## Disclaimer

We disclaim responsibility for user-generated content. The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities. It is prohibited for pornographic, violent and bloody content generation, and to generate content that is demeaning or harmful to people or their environment, culture, religion, etc. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards.
