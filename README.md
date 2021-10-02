# summer_2021_D
画像情報研究室2021年夏合宿が行われた際にD班で扱ったコードがまとまられたリポジトリです。  
お題はkaddleで2019年くらいに行われた[Gererative Dog Images](https://www.kaggle.com/c/generative-dog-images/data)という課題です。
 <br>

# Features
[こちらのリポジトリ](https://github.com/facebookresearch/pytorch_GAN_zoo)を基に合宿用に改良したものです。  
DCGAN, ProGAN, StyleGANを扱えるようになっています。   

# Requirement
numpy  
scipy  
visdom  
h5py  
nevergrad  
pytorch  
torchvision  
tqdm  
<br>

# Instration
```bash
git clone https://github.com/oki5656/summer_2021_D.git
```
 <br>

# Usage
1,config_dog120.jsonのpathDBを自分のデータセットのpathに書き換える  
pathDBの下の階層は以下を想定  
-all-dogs  
&emsp;-n02xxxxxx_nnnn.jpg  
-Annotation  
&emsp;-n02xxxxxx-犬種  
&emsp;&emsp;-n02xxxxxx_nnnn  
-all-dogs-bndbox.txt  

(※[txtファイルのリンク](https://drive.google.com/file/d/1Wp1OCAjiLlLpslsvwmvAFadm1IgT04p5/view))
<br>

2,学習
```bash
python train.py DCGAN -c config_dog120.json --restart -n dogs120 --np_vis
```
<br>

3,生成画像出力
```bash
python eval.py gen_vis_img -n dogs120 -m DCGAN
```
DCGANのところを適宜[StyleGAN, ProGAN]と書き換えるとそれに応じたネットワークのパラメータが生成に使用される。
 <br>

# Note
epochの設定パス<br>
```bash
models/trainer/standard_configuration/dcgan_config.py 
```
gereratorとdiscriminatorのlossのグラフ  
```bash
output_networks/loss_img
```
 

 
