import os
import shutil


def move_small_files(src_folder, dest_folder, size_limit_mb):
    # 将大小限制转换为字节
    size_limit_bytes = size_limit_mb * 1024 * 1024

    # 确保目标文件夹存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)

        # 检查是否是文件，并获取文件大小
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)

            # 如果文件小于指定大小，则移动
            if file_size < size_limit_bytes:
                shutil.move(file_path, os.path.join(dest_folder, filename))
                print(f"Moved file: {filename}")


if __name__ == "__main__":
    # 定义源文件夹和目标文件夹
    # source_folder = '/home/bd/project/Flare/FLARE24-Task3-MR/Training/AMOS_MR_good_spacing-833'
    # destination_folder = '/home/bd/project/Flare/FLARE24-Task3-MR/Training/AMOS_1'
    source_folder = '/home/bd/project/Flare/FLARE24-Task3-MR/Training/LLD-MMRI-3984'
    destination_folder = '/home/bd/project/Flare/FLARE24-Task3-MR/Training/LLD-1'

    # 设置文件大小限制（MB）
    size_limit = 1  # 1 MB

    # 执行文件转移
    move_small_files(source_folder, destination_folder, size_limit)
