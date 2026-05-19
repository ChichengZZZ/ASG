import socket
import struct
import time
# import math
# import torch
import numpy as np
np.set_printoptions(threshold=np.inf)

def get_current_time():
    """
    获取当前的时间
    """
    return time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

def array_to_binary(array, fmt='d'):
    """
    将array转换为二进制格式
    :param array: 要转换的array
    :param fmt: 二进制格式， 默认为'd'
    :return: 转换后的二进制格式
    """
    binary = struct.pack(fmt * len(array), *array)
    return binary


def binary_to_array(binary, fmt='d'):
    """
    将二进制格式转换为array
    :param binary: 要转换的二进制格式
    :param fmt: 二进制格式，默认为'd'
    :return: 转换后的array
    """
    array = struct.unpack(fmt * (len(binary) // struct.calcsize(fmt)), binary)
    return array


def binary_to_array_v2(binary_data):
    """
    将二进制格式转换为array（针对特定格式的二进制数据）
    :param binary_data: 要转换的二进制数据
    :return: 转换后的array
    """
    head = struct.unpack('i', binary_data[:4])[0]
    double_array = binary_to_array(binary_data[4:])
    return head, double_array


def vissim_states_process_traffic_flow(states, n_vehicle=5):
    """
    处理交通流状态数据
    :param states: 交通流状态数据，包含ego和其他车辆信息
    :param n_vehicle: 车辆数目，默认为5
    :return: 处理后的特征向量
    """
    states = np.array(states)
    lane_width, x, y, speed, angle, lane_offset = states[1], states[2], states[3], states[4], -states[5], states[6]
    print(f"Agent x:{x}, y:{y},speed:{speed}, angle:{angle}, lane offset:{lane_offset}")
    obs = states[2:].reshape(-1, 6)
    feature =[speed * np.cos(angle), speed * np.sin(angle)]
    ego_speed = [speed * np.cos(angle), speed * np.sin(angle)]

    mask = np.where(np.sum(obs[:, :3], axis=1) == 0, False, True)
    obs = obs[mask][1:, :].tolist()

    vehicles = [v for v in obs if np.linalg.norm([v[0] - x, v[1] - y]) <= 60]
    vehicles = sorted(vehicles, key=lambda v: np.linalg.norm([v[0] - x, v[1] - y]))
    vehicles = vehicles[:n_vehicle]
    print(vehicles)
    vehicle_num = len(vehicles)
    for i in range(len(vehicles)):

        veh_x = vehicles[i][0]
        veh_y = vehicles[i][1]
        veh_speed = speed - vehicles[i][2]
        veh_angle = -vehicles[i][3]
        print(f"Bv{i}: x:{veh_x}, y:{veh_y},speed:{veh_speed}, angle:{veh_angle}")
        feature.extend([x - veh_x, veh_y - y, ego_speed[0] - veh_speed * np.cos(veh_angle), ego_speed[1] - veh_speed * np.sin(veh_angle)])
    feature = np.array(feature)
    if len(vehicles) != n_vehicle:
        feature = np.concatenate((feature, np.zeros((n_vehicle - vehicle_num) * 4)), axis=0)

    return feature


def dispose_client_request(tcp_client_1, tcp_client_address, port=None, num_vehicle=10):
    """
    处理客户的请求数据
    :param tcp_client_1: 客户端套接字
    :param tcp_client_address: 客户端地址
    :param port: 端口（可选）
    :param num_vehicle: 车辆数目，默认为10
    """
    while True:
        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_data = tcp_client_1.recv(4096)
        if recv_data:
            head, states = binary_to_array_v2(recv_data)
            print("%s || data length is %d , head is %d" % (get_current_time(), len(states), head))

            if len(states) == 128:
                feature = None
                if (head // 10) == 0:
                    # traffic_flow_model
                    feature = vissim_states_process_traffic_flow(states, num_vehicle)
                elif (head // 10) == 1:
                    # adversarial_model
                    # feature = vissim_states_process_adv_model_d_13(states)
                    feature = vissim_states_process_traffic_flow(states, num_vehicle)

                print("TCP server got feature, which'length is  ", len(feature))
                print('input: ', np.array(feature))
                feature_binary = array_to_binary(feature)
                try:
                    print('请求：127.0.0.1：%d' % port)
                    sock.sendto(feature_binary, ('127.0.0.1', port))
                    data, server = sock.recvfrom(4096)
                    sock.close()
                    print('接收数据.')
                finally:
                    if head % 10 == 0:
                        # 连续控制
                        pass
                    elif head % 10 == 1:
                        # 离散控制
                        print("离散控制")
                        data = binary_to_array(data)
                        print('value is', data)
                        data = np.argmax(np.array(data))
                        print('actions is', data)
                        data = array_to_binary([data], 'i')
                        print('binary actions is', data)

                    tcp_client_1.sendall(data)
                    tcp_client_1.close()

                    print('process total time is %.4f s' % (time.time() - start))
                    # actions = binary_to_array(data)

                    # print('TCP server get ', actions)
            else:
                print('收到回信：', states)
                action_type_binary = array_to_binary([1], 'i')
                # actions = array_to_binary([0.0, 0.0])
                tcp_client_1.sendall(action_type_binary)
                tcp_client_1.close()
        print("关闭客户端 %s" % tcp_client_address[1])
        break


if __name__ == '__main__':
    a = [1.0, 2.0]
    a_ = binary_to_array(a)
    print(a_)
    b_ = array_to_binary(a_)
    print(b_)