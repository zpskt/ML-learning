# paddleocr 调优和本地部署
## 简介
私有化数据进行训练
教程地址：http://www.paddleocr.ai/main/version3.x/module_usage/text_recognition.html#_3
### 环境
paddlepaddle-gpu 3.2.0
paddleocr 3.4.0
python 3.10
### 训练流程
1. 创建环境并下载代码
```shell
conda create -n tea --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.10
conda activate tea
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
git checkout release/3.2
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
2. 准备数据集
示例数据集为：https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar       
可以自行下载然后参考
```shell
# 下载示例数据集
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar
tar -xf ocr_rec_dataset_examples.tar
mv ocr_rec_dataset_examples train_data
```

3. 下载预训练模型
```shell
# 下载 PP-OCRv5_server_rec 预训练模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
```

4. 训练
在此处注意事项，train.py是最新代码，可能和你安装的版本不一样，所以有的方法不可用，基本上看代码都可以对应修改
在PP-OCRv5_server_rec.yml中配置训练信息，包括训练数据集，模型保存路径，训练轮数，学习率等信息
```shell
# 单卡训练 (默认训练方式)
python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams
nohup python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams > output.log 2>&1 &
#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
        -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams
```

可以通过查看日志能够看到私有化训练后的模型会被存储到当前目录中
```shell
[2026/02/11 10:21:10] ppocr INFO: save model in ./output/PP-OCRv5_server_rec/latest
[2026/02/11 10:21:12] ppocr INFO: save model in ./output/PP-OCRv5_server_rec/iter_epoch_75
```
5. 模型导出
模型评估
```shell
python tools/eval.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o Global.pretrained_model=output/PP-OCRv5_server_rec/latest.pdparams
```
```shell
python tools/export_model.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o Global.pretrained_model=output/PP-OCRv5_server_rec/latest.pdparams Global.save_inference_dir="./PP-OCRv5_server_rec_infer/"
```
导出模型后，静态图模型会存放于当前目录的./PP-OCRv5_server_rec_infer/中，在该目录下，会看到如下文件：

./PP-OCRv5_server_rec_infer/
├── inference.json
├── inference.pdiparams
├── inference.yml

6. 使用私有训练后的模型
命令行使用
```shell
paddleocr text_recognition -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png --save_path ./output --model_dir ./PP-OCRv5_server_rec_infer
```
示例代码
```python
from paddleocr import TextRecognition
model = TextRecognition(model_name="PP-OCRv5_server_rec", model_dir='./PP-OCRv5_server_rec_infer')
output = model.predict(input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```
