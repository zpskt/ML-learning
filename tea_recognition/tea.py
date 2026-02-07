import base64
import os
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import json
import re


def cuttingImage_ai(image_url, ai_coordinates, output_filename=None):
    '''
    专门处理AI返回的[0, 999]归一化坐标

    :param image_url: 图片URL
    :param ai_coordinates: AI返回的坐标，如 [0, 142, 999, 912]
    :param output_filename: 输出文件名
    '''
    try:
        print(f"处理AI坐标: {ai_coordinates}")

        # 下载图片获取尺寸
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        width, height = image.size

        print(f"图片尺寸: {width}x{height}")

        # 将[0, 999]坐标转换为像素坐标
        x1 = int((ai_coordinates[0] / 999) * width)
        y1 = int((ai_coordinates[1] / 999) * height)
        x2 = int((ai_coordinates[2] / 999) * width)
        y2 = int((ai_coordinates[3] / 999) * height)

        print(f"转换后像素坐标: ({x1}, {y1}) -> ({x2}, {y2})")

        # 裁剪
        cropped = image.crop((x1, y1, x2, y2))

        # 保存
        if output_filename is None:
            output_filename = "ai_cropped.jpg"

        cropped.save(output_filename)
        print(f"✅ 保存: {output_filename}")

        return output_filename

    except Exception as e:
        print(f"❌ 错误: {e}")
        return None
#  编码函数： 将本地文件转换为 Base64 编码的字符串
# local_path = "D:/workspace/zpskt/ML-learning/tea_recognition/tea.jpeg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 将xxxx/eagle.png替换为你本地图像的绝对路径
# base64_image = encode_image(local_path)


def recognition(image_url):
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        # 添加环境变量DASHSCOPE_API_KEY 里面放上你的密钥即可
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 以下为北京地域的 base_url，若使用弗吉尼亚地域模型，需要将base_url换成https://dashscope-us.aliyuncs.com/compatible-mode/v1
        # 若使用新加坡地域的模型，需将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen3-vl-32b-instruct",  # 此处以qwen3-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
        messages=[
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "image_url",
                    #     # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    #     # PNG图像：  f"data:image/png;base64,{base64_image}"
                    #     # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                    #     # WEBP图像： f"data:image/webp;base64,{base64_image}"
                    #     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    # },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                    },
                    {"type": "text", "text": """
    你是一个专业的茶叶识别系统。请严格按照以下要求分析图片：

    ### 任务描述
    - **目标**：识别并返回茶叶的具体名称和类型，并提供茶叶在图片中的坐标信息。
    
    - **输入**：
      - 图片URL: `${image_url}`
    - **输出**：结果应以JSON格式返回，结构如下：
      ```json
      {
        "teaName": "${tea_name}",
        "teaType": "${tea_type}",
        "coordinates": [${x1},${y1},${x2},${y2}]
        "size": "image_size" //返回识别图片的像素
      }
      ```

    ### 注意事项
    1. **图片质量**：请确保提供的原始图片足够清晰，以便于准确地进行茶叶识别。
    2. **准确性**：返回的结果应当尽可能精确地反映图片中的茶叶特征。
    3. **处理其他物体**：如果图片中除了茶叶外还有其他物体，请只处理和返回与茶叶直接相关的信息，并且将茶叶的坐标返回。这些坐标将用于后续的切图操作。
    4. **坐标规则（非常重要！）**：
坐标含义：
x1: 左上角X坐标（必须 >= 0）

y1: 左上角Y坐标（必须 >= 0）

x2: 右下角X坐标（必须 <= 图片宽度）

y2: 右下角Y坐标（必须 <= 图片高度）

坐标约束：

x1 < x2, y1 < y2

所有坐标必须是整数

坐标值必须在图片实际尺寸范围内

示例：

如果图片尺寸是700x1200，有效坐标可能是：[50, 100, 650, 800]

绝对不允许：[0, 142, 999, 912]（因为999 > 700）
    通过遵循上述指导原则，你可以帮助用户更准确地识别茶叶及其相关信息。
     """},
                ],
            },
        ],
    )
    # 获取AI返回的内容
    response_content = completion.choices[0].message.content
    print("AI识别结果:")
    print(response_content)
    print(type(response_content))
    # 提取JSON部分
    json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 如果没有代码块格式，尝试直接解析
        json_str = response_content.strip()

    # 解析JSON
    result = json.loads(json_str)
    return result

if __name__ == '__main__':


    # 解析JSON响应
    try:
        image_url = "https://tse3-mm.cn.bing.net/th/id/OIP-C.-BvOIK32se-HMSVwHLHI3wHaEJ?w=285&h=180&c=7&r=0&o=7&pid=1.7&rm=3"
        response = requests.get(image_url, stream=True, timeout=5)
        image = Image.open(BytesIO(response.content))
        true_width, true_height = image.size
        print(f"📏 真实图片尺寸: {true_width}x{true_height}")

        result = recognition(image_url)

        # 提取坐标
        coordinates = result.get('coordinates', [])
        if len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            tea_name = result.get('teaName', 'unknown_tea')
            tea_type = result.get('teaType', 'unknown_type')
            
            print(f"\n提取到的茶叶信息:")
            print(f"茶叶名称: {tea_name}")
            print(f"茶叶类型: {tea_type}")
            print(f"坐标位置: ({x1}, {y1}) to ({x2}, {y2})")
            
            # 执行切图
            output_filename = f"{tea_name}_{tea_type}.jpg"
            # 坐标位置为[0, 143, 1000, 912]
            saved_path = cuttingImage_ai(image_url,coordinates,output_filename)
            if saved_path:
                print(f"\n✅ 切图成功！保存路径: {saved_path}")
            else:
                print("\n❌ 切图失败！")
        else:
            print("❌ 未找到有效的坐标信息")
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        print("原始响应内容:")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")