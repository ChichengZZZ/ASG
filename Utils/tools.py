from os import path
import os
import json


# 定义一个函数，返回 assets 文件夹的绝对路径
def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../data/assets'))
    # 使用 path.abspath 获取当前脚本的绝对路径，并使用 path.dirname 获取该路径的父目录
    # 然后使用 path.join 将父目录路径与 '../data/assets' 进行合并
    # 最后再次使用 path.abspath 获取合并后的绝对路径，并返回

# 定义一个函数，用于将数据保存为 JSON 文件
def save(data:list, path:str):
    if not path.endswith('json'):
        path = path + ".json"
        # 如果传入的路径不是以 '.json' 结尾，则将其加上 '.json' 后缀

    json_data = json.dumps(data)
    # 使用 json.dumps 将数据转换为 JSON 格式的字符串

    with open(path, "w") as f:
        f.write(json_data)
        # 将 JSON 字符串写入指定路径的文件中


# 定义一个函数，用于从 JSON 文件中加载数据
def load(path: str):

    if not path.endswith('json'):
        path = path + ".json"
        # 如果传入的路径不是以 '.json' 结尾，则将其加上 '.json' 后缀

    if os.path.exists(path):
        # 判断路径是否存在

        with open(path, "r") as f1:
            orin_Data = f1.read()
            # 读取文件内容
            orin_Data = json.loads(orin_Data)
            # 使用 json.loads 将文件内容转换为 Python 对象
            return orin_Data
            # 返回加载的数据
    else:
        raise 'Path {} Not Found.'.format(path)
        # 如果路径不存在，则抛出异常，提示路径不存在