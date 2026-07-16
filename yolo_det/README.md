#### 创建conda环境（推荐）

```bash
conda create -n yolo --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.9
conda activate yolo
#pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -U ultralytics -i https://mirrors.aliyun.com/pypi/simple/

pip install torch torchvision -i https://mirrors.aliyun.com/pypi/simple/ #macos用户

```