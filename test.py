from PIL import Image
import os

def convert_to_grayscale(input_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 打开图像并转为灰度图
        img = Image.open(input_path).convert('L')

        # 保存转换后的图像
        img.save(output_path)

if __name__ == "__main__":
    input_directory = "F:\pycharm\data\images\images\DSUnet\\test\images\\test\pred_sel"
    output_directory = "F:\pycharm\data\images\images\DSUnet\\test\images\\test\preds_sel"

    convert_to_grayscale(input_directory, output_directory)
