# Text2Reward: Automated Dense Reward Function Generation for Reinforcement Learning

<p align="left">
    <a href="https://img.shields.io/badge/PRs-Welcome-red">
        <img src="https://img.shields.io/badge/PRs-Welcome-red">
    </a>
    <a href="https://img.shields.io/github/last-commit/xlang-ai/text2reward?color=green">
        <img src="https://img.shields.io/github/last-commit/xlang-ai/text2reward?color=green">
    </a>
    <br/>
</p>

Code for paper [Text2Reward: Automated Dense Reward Function Generation for Reinforcement Learning](https://arxiv.org/abs/2309.11489).
Please refer to our [project page](https://text-to-reward.github.io/) for more demonstrations and up-to-date related resources. 


## Updates
- **2023-10-09**: We released our [code](https://github.com/xlang-ai/text2reward).
- **2023-09-20**: We release the [paper](https://arxiv.org/abs/2309.11489) and [website](https://text-to-reward.github.io/) of text2reward.


## Dependencies
To establish the environment, run this code in the shell:
```shell
conda create -n text2reward python=3.7
conda activate text2reward
cd ManiSkill2
pip install -e .
pip install stable-baselines3==1.8.0 wandb tensorboard
bash download_data.sh
cd ..
cd Metaworld
pip install -e .
```

## TroubleShooting

1. If you have not installed `mujoco` yet, please follow the instructions from [here](https://github.com/openai/mujoco-py#install-mujoco) to install it. After that, please try the following commands to confirm the successful installation:

```shell
$ python3
>>> import mujoco_py
```

2. If you encounter the following errors when running ManiSkill2, we refer you to read the documents [here](https://haosulab.github.io/ManiSkill2/getting_started/installation.html#vulkan).
   - `RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed`
   - `Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.`
   - `Segmentation fault (core dumped)`

## Usage

### Reimplement

To reimplement our experiment results, you can run the following scripts:

**ManiSkill2:**

```shell
bash run_oracle.sh
bash run_zero_shot.sh
bash run_few_shot.sh
```

It's normal to encounter the following warnings:

```shell
[svulkan2] [error] GLFW error: X11: The DISPLAY environment variable is missing
[svulkan2] [warning] Continue without GLFW.
```

**MetaWorld:**

```shell
bash run_oracle.sh
bash run_zero_shot.sh
```

### Run your own experiment

By default, the `run_oracle.sh` script above uses the expert-written rewards provided by the environment; the `run_zero_shot.sh` and `run_few_shot.sh` scripts use the generated rewards used in our experiments. If you want to run a new experiment based on the reward you provide, just follow the bash script above and modify the `--reward_path` parameter to the path of your own reward.

## Citation

If you find our work helpful, please cite us:

```bibtex
@article{text2reward,
  title={Text2Reward: Automated Dense Reward Function Generation for Reinforcement Learning},
  author={Xie, Tianbao and Zhao, Siheng and Wu, Chen Henry and Liu, Yitao and Luo, Qian and Zhong, Victor and Yang, Yanchao and Yu, Tao},
  journal={arXiv preprint arXiv:2309.11489},
  year={2023}
}
```

## Contributors
<a href="https://github.com/Timothyxxx">  <img src="https://avatars.githubusercontent.com/u/47296835?v=4"  width="50" /></a>
<a href="https://github.com/Hilbert-Johnson">  <img src="https://avatars.githubusercontent.com/u/77528902?v=4"  width="50" /></a>
<a href="https://github.com/ChenWu98"><img src="https://avatars.githubusercontent.com/u/28187501?v=4"  width="50" /></a>
<a href="https://github.com/taogoddd">  <img src="https://avatars.githubusercontent.com/u/98326623?v=4"  width="50" /></a>
<a href="https://qianluo.netlify.app/"><img src="https://qianluo.netlify.app/author/qian-luo/avatar_hu5e5a95a93d56ec8418a5a4471effb4fb_2337021_270x270_fill_q75_lanczos_center.jpg"  width="50" /></a>
