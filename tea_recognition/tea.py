import base64
import os
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import json
import re


def cuttingImage(image_url, x1, y1, x2, y2, output_filename=None, auto_fix=True, verbose=True):
    '''
    改进的切图函数：自动处理坐标问题

    :param image_url: 图片的URL地址
    :param x1: 裁剪区域左上角x坐标
    :param y1: 裁剪区域左上角y坐标
    :param x2: 裁剪区域右下角x坐标
    :param y2: 裁剪区域右下角y坐标
    :param output_filename: 输出文件名
    :param auto_fix: 是否自动修正超出边界的坐标
    :param verbose: 是否显示详细信息
    :return: 保存的文件路径或错误信息
    '''
    try:
        if verbose:
            print(f"📥 正在下载图片: {image_url}")

        # 下载图片
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        # 打开图片
        image = Image.open(BytesIO(response.content))
        img_width, img_height = image.size

        if verbose:
            print(f"📐 图片实际尺寸: {img_width} x {img_height}")
            print(f"📍 请求坐标: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"📏 请求区域: {x2 - x1} x {y2 - y1}")

        # 记录原始坐标
        original_coords = (x1, y1, x2, y2)

        # 检查坐标问题
        issues = []
        if x1 < 0: issues.append(f"x1({x1}) < 0")
        if y1 < 0: issues.append(f"y1({y1}) < 0")
        if x2 > img_width: issues.append(f"x2({x2}) > 宽度({img_width})")
        if y2 > img_height: issues.append(f"y2({y2}) > 高度({img_height})")
        if x1 >= x2: issues.append(f"x1({x1}) >= x2({x2})")
        if y1 >= y2: issues.append(f"y1({y1}) >= y2({y2})")

        if issues:
            if verbose:
                print(f"⚠️  发现{len(issues)}个问题: {', '.join(issues)}")

            if not auto_fix:
                raise ValueError(f"坐标无效: {', '.join(issues)}")

            # 自动修正坐标
            if verbose:
                print("🔧 开始自动修正坐标...")

            # 修正负值
            x1 = max(0, x1)
            y1 = max(0, y1)

            # 修正超出边界的值
            x2 = min(x2, img_width)
            y2 = min(y2, img_height)

            # 确保x1 < x2, y1 < y2
            if x1 >= x2:
                if x1 >= img_width - 10:  # 如果x1在右边边界
                    x1 = max(0, img_width - 200)  # 向左移动200像素
                x2 = min(img_width, x1 + 100)  # 设置最小宽度100像素
                if verbose:
                    print(f"  修正x坐标: x1={x1}, x2={x2}")

            if y1 >= y2:
                if y1 >= img_height - 10:  # 如果y1在底部边界
                    y1 = max(0, img_height - 200)  # 向上移动200像素
                y2 = min(img_height, y1 + 150)  # 设置最小高度150像素
                if verbose:
                    print(f"  修正y坐标: y1={y1}, y2={y2}")

            # 确保最小裁剪尺寸
            min_width, min_height = 50, 50
            if (x2 - x1) < min_width:
                x2 = min(img_width, x1 + min_width)
                if verbose:
                    print(f"  宽度过小，调整为: {x2 - x1}像素")

            if (y2 - y1) < min_height:
                y2 = min(img_height, y1 + min_height)
                if verbose:
                    print(f"  高度过小，调整为: {y2 - y1}像素")

            # 检查修正后的坐标
            if x1 >= x2 or y1 >= y2:
                # 如果修正后仍然无效，使用智能默认值
                if verbose:
                    print("⚠️  修正后坐标仍然无效，使用智能默认值")

                # 根据图片尺寸设置合理的裁剪区域
                if img_width >= 500 and img_height >= 500:
                    # 大图片：裁剪中心70%区域
                    margin_w = img_width // 6
                    margin_h = img_height // 6
                    x1, y1 = margin_w, margin_h
                    x2, y2 = img_width - margin_w, img_height - margin_h
                else:
                    # 小图片：裁剪中心80%区域
                    margin_w = img_width // 10
                    margin_h = img_height // 10
                    x1, y1 = margin_w, margin_h
                    x2, y2 = img_width - margin_w, img_height - margin_h

        if verbose:
            print(f"✅ 最终坐标: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"📏 裁剪区域: {x2 - x1} x {y2 - y1}")

        # 裁剪图片
        cropped_image = image.crop((x1, y1, x2, y2))

        # 设置默认文件名
        if output_filename is None:
            import time
            timestamp = int(time.time())
            output_filename = f"cropped_tea_{timestamp}.jpg"

        # 保存裁剪后的图片
        cropped_image.save(output_filename)

        if verbose:
            print(f"✅ 切图完成，已保存到: {output_filename}")
            print(f"📊 裁剪后尺寸: {cropped_image.size}")

        # 返回详细信息
        return {
            "file_path": output_filename,
            "original_coords": original_coords,
            "final_coords": (x1, y1, x2, y2),
            "image_size": (img_width, img_height),
            "crop_size": cropped_image.size,
            "auto_fixed": len(issues) > 0,
            "issues_found": issues
        }

    except requests.RequestException as e:
        error_msg = f"下载图片失败: {e}"
        if verbose:
            print(f"❌ {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"切图过程出错: {e}"
        if verbose:
            print(f"❌ {error_msg}")
        return {"error": error_msg}
#  编码函数： 将本地文件转换为 Base64 编码的字符串
local_path = "D:/workspace/zpskt/ML-learning/tea_recognition/tea.jpeg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 将xxxx/eagle.png替换为你本地图像的绝对路径
base64_image = encode_image(local_path)


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
        model="qwen3-vl-plus",  # 此处以qwen3-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
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
            saved_path = cuttingImage(image_url, x1, y1, x2, y2, output_filename)
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