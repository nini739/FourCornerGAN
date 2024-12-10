from PIL import Image, ImageDraw, ImageFont
import os
import json
import unicodedata

def generate_hanzi_images(ttf_file, label_file, output_dir, image_size=(128, 128), font_size=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    font = ImageFont.truetype(ttf_file, font_size)

    with open(label_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    for label in labels:
        hanzi = label['hanzi']
        image = Image.new('RGB', image_size, (255, 255, 255))
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), hanzi, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (image_size[0] - w) / 2 - bbox[0]
        y = (image_size[1] - h) / 2 - bbox[1]
        draw.text((x, y), hanzi, font=font, fill=(0, 0, 0))
        image.save(os.path.join(output_dir, f'{ord(hanzi):04X}.png'))

# 示例使用
ttf_file = '.ttf'  # 替换为您的TTF文件路径
label_file = 'hanzi_four_corner_onehot.json'  # 替换为您的标签文件路径
output_dir = ''  # 替换为您的输出目录

generate_hanzi_images(ttf_file, label_file, output_dir, image_size=(128, 128), font_size=100)