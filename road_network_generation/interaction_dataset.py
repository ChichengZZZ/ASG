import os
import sys
current_dir = sys.path[0].replace("\\", "/")
project_dir = os.sep.join(current_dir.split('/')[:-1]).replace("\\", "/")
sys.path.append(project_dir)
import json
import numpy.linalg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import os
import gc
from sklearn.cluster import KMeans, k_means
from road_network_generation import GeneratedRoadNet


class InterAction(GeneratedRoadNet):

    def __init__(self, config: dict):
        config["sample_point_num"] = 5
        config['crossing_center_num'] = [4, 5, 6]
        config['coordinate_name'] = ['x', 'y']
        config["vehicle_id_name"] = "track_id"
        config['location'] = ['TC_BGR_Intersection_VA', 'USA_Intersection_MA', 'USA_Intersection_GL']
        config['road_net_categories_num'] = len(config['location'])
        super(InterAction, self).__init__(config)

    def parse_dataset_path(self) -> None:
        """
        定义了一个内部函数get_all_foler_path(path: str)，用于获取指定路径path下包含"Intersection"关键字的文件夹路径。
        - 将获取到的文件夹路径按字母顺序排序。
        - 将获取到的文件夹路径存储在self.data_path中。
        """
        # 提取所有包含Intersection关键字的文件夹
        def get_all_foler_path(path: str):
            folder_name = []
            if not osp.isdir(path):
                return folder_name

            for n in os.listdir(path):
                file_path = osp.join(path, n).replace("\\", "/")
                if n.startswith("vehicle") and n.endswith("csv") and "Intersection" in file_path.split("/")[-2]:
                    folder_name.append(file_path)
                else:
                    folder_name.extend(get_all_foler_path(file_path))

            return folder_name

        self.data_path = sorted(get_all_foler_path(self.config["dataset_path"]))

    def classify_path(self) -> None:
        for cls_idx, lo in enumerate(self.config['location']):
            data = []
            for p in self.data_path:
                if lo in p:
                    data.append(p)
                    self.class_path_reverse[p] = cls_idx
            else:
                self.class_path[cls_idx] = data

    def read_data(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        out_df = None
        for i, name in enumerate(self.config['agent_type']):
            if i == 0:
                out_df = self.separate_data_by_key_value(data, self.config['agent_type_name'], name)
            else:
                out_df = pd.concat([out_df,
                                     self.separate_data_by_key_value(data, self.config['agent_type_name'], name)])
        return out_df

    def save_road(self, roads: dict, idx: int):
        line_type = dict()
        for line_key in list(roads.keys()):
            if idx == 0:
                if line_type in ["(0, 2)", "(2, 1)"]:
                    line_type[line_key] = ["n", "n"]
                else:
                    line_type[line_key] = ["c", "c"]
            elif idx == 1:
                pass
            elif idx == 2:
                pass

        path = os.path.join(self.config["output_path"], f"{self.config['location'][idx]}.json")
        print(f"save data in {path}.")
        roads["line_type"] = line_type
        json_data = json.dumps(roads)
        with open(path, "w") as f:
            f.write(json_data)


if __name__ == "__main__":

    config = {
        "dataset_path": f"{project_dir}/data/datasets/Interaction",
        "output_path": f"{project_dir}/data/output",
        "save_road_kwargs": {
        }
    }
    dataset = InterAction(config)
    dataset.run(0)
