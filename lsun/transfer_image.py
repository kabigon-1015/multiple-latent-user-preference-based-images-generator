import os
import shutil

ROOT_PATH = "living_room_train_img"
MOVED_PATH = "2"
count = 0

def process(file_path):
    global count
    shutil.move(file_path, MOVED_PATH)
    count += 1

def recursive_file_check(filepath):
    global count
    if count > 30000:
        exit(0)
    if os.path.isdir(filepath):
        # directoryだったら中のファイルに対して再帰的にこの関数を実行
        files = os.listdir(filepath)
        for file in files:
            if file != ".DS_Store":
                recursive_file_check(filepath + "/" + file)
    else:
        # fileだったら処理
        process(filepath)

if __name__ == '__main__':
    recursive_file_check(ROOT_PATH)