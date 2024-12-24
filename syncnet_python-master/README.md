# test-audio2head

## 使用docker进行部署
### 首先下载checkpoints

首先请从以下网址下载[google-drive](https://drive.google.com/file/d/1tvI43ZIrnx9Ti2TpFiEO4dK5DOwcECD7/view?usp=sharing)，当上以连接无法访问或者载入后无法运行时，请下载这个备用[checkpoint2](https://drive.google.com/drive/folders/1k-6im7e4EkPjQSXCO7jWEQwYSHCsCyJb?usp=sharing)。

下载完成后，请将下载好的checkpoints文件夹放到syncnet_python-master/Audio2Head/Audio2Head目录下

完成以上步骤后，请用以下命令构建docker容器
``` bash
docker build -t syn .
```

构建完成后，你可以开始运行容器，但是请注意，请确保你准备好了用于的评估的原视频，用于模型生成视频的评估视频的第一帧图片和音频。同时还需要注意，你的评估视频，用于模型生成视频的第一帧图片和音频的文件名应该相同。（比如，你的一个评估视频是eric.mp4，那么你应该截取这个视频的第一帧图片用于模型生成，并命名为eric.png，同截取这个视频的音频，命名为eric.wav）然后将你的所有评估视频放一个文件夹（例如ref_video），所有图片放一个文件夹（例如input_img），所有音频放一个文件夹（例如input_wav）然后通过以下命令进行挂载和运行程序
```bash
docker run --rm --gpus all \
-v /path/to/your/input_img/:/app/Audio2Head/Audio2Head/input_img \
-v /path/to/your/input_wav/:/app/Audio2Head/Audio2Head/input_wav \
-v /path/to/your/ref_video/:/app/ref_video \
syn
```

然后通过批处理程序，会自动生成调用模型生成视频然后进行评估
## 使用conda环境部署
如果你的系统没有Anconda环境，可以按以下操作
### 1.首先安装conda环境
打开终端，下载 Miniconda 安装包：
``` bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

```
运行安装脚本(注意最后添加环境变量时选择yes，全部输yes)
```
bash Miniconda3-latest-Linux-x86_64.sh
```
完成安装后，执行：
```
source ~/.bashrc
```

### 2.创建conda环境并进入

```
conda create -n syn python=3.10 && conda activate syn
```

### 3.进入到项目目录
首先安装依赖

```
# 确保你的电脑安装了cuda12以上的版本
pip install -r requirements.txt
pip install imageio[ffmpeg]
pip install cupy-cuda12x
apt-get install ffmpeg
```
然后将你的所有评估视频的第一帧图像放到
```
Audio2Head\Audio2Head\input_img
```
将所有评估视频的音频放到
```
Audio2Head\Audio2Head\input_wav
```
注意，input_wav里要求wav格式，同时input_img和input_wav对应的视频名字应该相同

然后将你的评估视频保存到
```
ref_video
```

然后运行run_inference.sh批处理文件
```
./run_inference.sh
```
等上一个批处理脚本运行完后，运行
```
python batch_psnr.py
```
开始评估
