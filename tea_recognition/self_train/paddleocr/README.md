# paddleocr 再训练
## 简介
私有化数据进行训练
教程地址：http://www.paddleocr.ai/main/version3.x/module_usage/text_recognition.html#_3
### 训练流程
1. 准备数据集
示例数据集为：https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar       
可以自行下载然后参考
```shell
# 下载示例数据集
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar
tar -xf ocr_rec_dataset_examples.tar
```
2. 下载预训练模型
```shell
# 下载 PP-OCRv5_server_rec 预训练模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
```
3. 下载代码
```shell
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```
4. 训练
```shell
#单卡训练 (默认训练方式)
python3 tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
   -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
        -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams
```self_train/paddleocr/README.md
