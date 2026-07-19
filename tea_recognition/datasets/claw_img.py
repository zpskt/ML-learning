#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import requests
import re
import time
from urllib import parse


class BaiduImageHTMLSpider:
    def __init__(self):
        self.image_counter = 0
        # 真实的浏览器请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
        }

    def create_directory(self, keyword):
        """创建保存图片的目录"""
        self.directory = os.path.join(os.getcwd(), keyword)
        os.makedirs(self.directory, exist_ok=True)
        print(f"图片将保存至: {self.directory}")

    def get_image_urls_from_html(self, keyword, pn=0):
        """
        从百度图片搜索的HTML页面中提取图片URL
        pn: 分页参数，0表示第一页，30表示第二页，以此类推
        """
        # 构建搜索URL
        encoded_keyword = parse.quote(keyword)
        url = f'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=index&fr=&hs=0&sf=1&fmq=&pv=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word={encoded_keyword}&pn={pn}'

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            html_content = response.text

            # 方法1: 从 "objURL" 字段提取图片链接 (最常见)
            # 匹配类似: "objURL":"http://xxx.jpg"
            pattern_obj = r'"objURL":"(https?://[^"]+\.(jpg|jpeg|png|gif|webp))"'
            urls_obj = re.findall(pattern_obj, html_content, re.I)
            urls = [url[0] for url in urls_obj]

            # 如果objURL提取不到，尝试提取thumbURL
            if not urls:
                pattern_thumb = r'"thumbURL":"(https?://[^"]+\.(jpg|jpeg|png|gif|webp))"'
                urls_thumb = re.findall(pattern_thumb, html_content, re.I)
                urls = [url[0] for url in urls_thumb]

            # 如果还提取不到，尝试最通用的图片链接匹配
            if not urls:
                # 匹配类似: "https://xxx.jpg" 或 "http://xxx.png"
                pattern_general = r'"(https?://[^"]+\.(jpg|jpeg|png|gif|webp))"'
                all_urls = re.findall(pattern_general, html_content, re.I)
                # 过滤掉一些明显不是图片的链接
                filtered = []
                for url, ext in all_urls:
                    if 'baidu.com' not in url and 'static' not in url and 'logo' not in url:
                        filtered.append(url)
                urls = filtered

            # 去重
            urls = list(dict.fromkeys(urls))
            print(f"  从第 {pn // 30 + 1} 页提取到 {len(urls)} 个图片链接")
            return urls

        except Exception as e:
            print(f"请求或解析失败: {e}")
            return []

    def convert_to_real_url(self, img_url):
        """
        将百度图片的缩略图URL转换为真实图片URL
        """
        # 如果是百度缩略图格式 (通常包含 'thumbnail' 或 'thumb' 字样)
        if 'thumbnail' in img_url or 'thumb' in img_url:
            # 尝试替换为 'large' 或 'download' 获取原图
            real_url = img_url.replace('thumbnail', 'large').replace('thumb', 'download')
            return real_url

        # 如果是 "http://p1.*.com/..." 格式，通常可以直接使用
        if img_url.startswith('http://p') or img_url.startswith('https://p'):
            return img_url

        # 其他情况，尝试移除可能的后缀参数
        if '?' in img_url:
            return img_url.split('?')[0]

        return img_url

    def is_valid_image(self, filepath):
        """
        检查文件是否有效的图片文件
        """
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)
                # JPEG: FF D8 FF
                if header[:3] == b'\xff\xd8\xff':
                    return True
                # PNG: 89 50 4E 47
                if header[:4] == b'\x89PNG':
                    return True
                # GIF: 47 49 46 38
                if header[:4] == b'GIF8':
                    return True
                # WebP: 52 49 46 46
                if header[:4] == b'RIFF':
                    return True
                # BMP: 42 4D
                if header[:2] == b'BM':
                    return True
            return False
        except:
            return False

    def download_image(self, img_url, filepath):
        """
        下载单张图片 - 增强版
        """
        # 先尝试转换为真实URL
        real_url = self.convert_to_real_url(img_url)

        # 准备多个不同的请求头，模拟不同来源
        headers_list = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://image.baidu.com/',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://www.baidu.com/',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://image.baidu.com/search/index',
            }
        ]

        # 尝试用不同的请求头下载
        for headers in headers_list:
            try:
                response = requests.get(real_url, headers=headers, timeout=15, stream=True)

                # 如果状态码是403或404，可能是防盗链，尝试下一个headers
                if response.status_code in [403, 404]:
                    continue

                response.raise_for_status()

                # 检查文件大小，如果太小可能是缩略图，但先下载看看
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) < 1024:  # 小于1KB可能是错误页
                    continue

                content_type = response.headers.get('content-type', '')
                if 'image' not in content_type:
                    continue

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                # 检查下载的文件是否有效图片（简单检查文件头）
                if self.is_valid_image(filepath):
                    return True
                else:
                    # 如果文件无效，删除它
                    os.remove(filepath)
                    return False

            except Exception as e:
                continue

        # 如果所有方法都失败，尝试直接使用原始URL（不转换）
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://image.baidu.com/',
            }
            response = requests.get(img_url, headers=headers, timeout=15)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
        except:
            return False

    def crawl(self, keyword, target_count=150):
        """主爬取函数"""
        self.create_directory(keyword)
        print(f"开始搜索关键词: {keyword}")

        pn = 0  # 分页起始
        max_pages = 10  # 最多爬取页数

        while self.image_counter < target_count and max_pages > 0:
            print(f"\n正在解析第 {pn // 30 + 1} 页...")
            img_urls = self.get_image_urls_from_html(keyword, pn=pn)

            if not img_urls:
                print("  本页没有提取到图片链接，尝试翻页...")
            else:
                for idx, url in enumerate(img_urls):
                    if self.image_counter >= target_count:
                        break
                    filename = os.path.join(self.directory, f"{self.image_counter:05d}.jpg")
                    print(f"  下载第 {self.image_counter + 1} 张...", end=' ')
                    if self.download_image(url, filename):
                        self.image_counter += 1
                        print("✅")
                    else:
                        print("❌")
                    time.sleep(0.3)

            # 翻页
            pn += 30
            max_pages -= 1
            time.sleep(0.5)

        print(f"\n🎉 完成！共下载 {self.image_counter} 张图片，保存在: {self.directory}")


if __name__ == '__main__':
    spider = BaiduImageHTMLSpider()

    keyword = input("请输入搜索关键词 (例如: 普洱茶包装): ").strip()
    if not keyword:
        keyword = "茶叶包装"

    # 先下载少量测试
    spider.crawl(keyword, target_count=50)