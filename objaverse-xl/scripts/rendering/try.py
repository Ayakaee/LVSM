from PIL import Image

# 打开 PNG 图片
image = Image.open("objaverse-xl/scripts/rendering/data/850a67b08cf54c8fa9d1d190f37eef6d/000.png")

# 获取图片的分辨率（宽度和高度）
width, height = image.size

print(f"图片分辨率: {width}x{height}")