import logging
import socket

import numpy as np


class ValveTest:
    def __init__(self):
        self.last_cmd = None
        self.reminder = """======================================================================================
快，给我个指令😉😉😉︎：
a. 开始命令 st.                                      e. 设置 光谱(a)相机 的延时，格式 e,500
b. 停止命令 sp.                                      f. 设置 彩色(b)相机 的延时, 格式 f,500
c. 设置光谱相机分频系数,4的倍数且>=8, 格式 c,8           g. 发个da和db完全重叠的mask
d. 阀板的脉冲分频系数,>=2即可                          h. 发个da和db呈现出X形的mask

你给我个小于256的数字，我就测试对应的喷阀。如果已经测试过一个，可以直接回车测下一个。
给q指令我就退出。
======================================================================================\n"""
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
        host = socket.gethostname()  # 获取本地主机名
        port = 13452  # 设置端口
        self.s.bind((host, port))  # 绑定端口
        self.s.listen(1)  # 等待客户端连接
        self.c = None

    def run(self):
        print("我在等连接...")
        self.c, addr = self.s.accept()  # 建立客户端连接
        print('和它的链接建立成功了：', addr)
        while True:
            value = input(self.reminder)
            if value == 'q':
                print("好的，我退出啦")
                break
            else:
                self.process_cmd(value)
        self.c.close()  # 关闭连接

    @staticmethod
    def cmd_padding(cmd):
        return b'\xAA' + cmd + b'\xFF\xFF\xBB'

    @staticmethod
    def param_cmd_parser(cmd, default_value, checker=None):
        try:
            value = int(cmd.split(',')[-1])
        except:
            print(f'你给的值不对啊，我先给你弄个{default_value}吧')
            value = default_value
        if checker is not None:
            if not checker(value):
                return None
        return value

    def process_cmd(self, value):
        if value == 'a':
            # a.开始命令
            cmd = b'\x00\x03' + 'sa'.encode('ascii') + b'\xFF'
        elif value == 'b':
            # b.停止命令
            cmd = b'\x00\x03' + 'sb'.encode('ascii') + b'\xFF'
        elif value.startswith('c'):
            # c. 设置光谱相机分频，得是4的倍数而且>=8，格式：c,8
            checker = lambda x: (x % 4 == 0) and (x >= 8)
            value = self.param_cmd_parser(value, default_value=8, checker=checker)
            if value is None:
                print("值需要是4的倍数且大于8")
                return
            cmd = b'\x00\x0a' + 'sc'.encode('ascii') + f"{value:08d}".encode('ascii')
        elif value.startswith('d'):
            # d. 阀板的脉冲分频系数，>=2即可
            checker = lambda x: x >= 2
            value = self.param_cmd_parser(value, default_value=2, checker=checker)
            if value is None:
                print("你得大于等于2")
                return
            cmd = b'\x00\x0a' + 'sv'.encode('ascii') + f"{value:08d}".encode('ascii')
        elif value.startswith('e'):
            # e. 设置 光谱(a)相机 的延时，格式 e,500
            checker = lambda x: (x >= 0)
            value = self.param_cmd_parser(value, default_value=2, checker=checker)
            if value is None:
                print("你得大于等于0")
                return
            cmd = b'\x00\x0a' + 'sa'.encode('ascii') + f"{value:08d}".encode('ascii')
        elif value.startswith('f'):
            # f. 设置 RGB(b)相机 的延时，格式 e,500
            checker = lambda x: (x >= 0)
            value = self.param_cmd_parser(value, default_value=2, checker=checker)
            if value is None:
                print("你得大于等于0")
                return
            cmd = b'\x00\x0a' + 'sb'.encode('ascii') + f"{value:08d}".encode('ascii')
        elif value == 'g':
            # g.发个da和db完全重叠的mask
            mask_a, mask_b = np.eye(256, dtype=np.uint8), np.eye(256, dtype=np.uint8)
            len_a, data_a = self.format_data(mask_a)
            len_b, data_b = self.format_data(mask_b)
            cmd = len_a + 'da'.encode('ascii') + data_a
            self.send(cmd)
            cmd = len_b + 'db'.encode('ascii') + data_b
        elif value == 'h':
            # h.发个da和db呈现出X形的mask
            mask_a, mask_b = np.eye(256, dtype=np.uint8), np.eye(256, dtype=np.uint8).T
            len_a, data_a = self.format_data(mask_a)
            len_b, data_b = self.format_data(mask_b)
            cmd = len_a + 'da'.encode('ascii') + data_a
            self.send(cmd)
            cmd = len_b + 'db'.encode('ascii') + data_b
        elif value == '' and self.last_cmd is not None:
            self.last_cmd += 1
            if self.last_cmd > 256:
                self.last_cmd = 1
            print(f'自动变化到 喷阀测试 {self.last_cmd}')
            value = self.last_cmd
            cmd = b'\x00\x0A' + 'te'.encode('ascii') + f"{value:08d}".encode('ascii')
        else:
            try:
                value = int(value)
            except Exception as e:
                print(e)
                print(f"你给的指令: {value} 咋看都不对")
                return
            if (value <= 256) and (value >= 1):
                cmd = b'\x00\x0A' + 'te'.encode('ascii') + f"{value:08d}".encode('ascii')
                self.last_cmd = value
            else:
                print(f'你给的指令: {value} 值不对，我们有256个阀门, 范围是 [1, 256]')
                return
        self.send(cmd)

    def send(self, cmd: bytes) -> None:
        cmd = self.cmd_padding(cmd)
        print("我要 send 这个了:")
        print(cmd.hex())
        try:
            self.c.send(cmd)
        except Exception as e:
            print(f"发失败了, 这是我找到的错误信息:\n{e}")
            return
        print("发好了")

    @staticmethod
    def format_data(array_to_send: np.ndarray) -> (bytes, bytes):
        data = np.packbits(array_to_send, axis=-1)
        data = data.tobytes()
        data_len = (len(data) + 2).to_bytes(2, 'big')
        return data_len, data


class VirtualValve:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 声明socket类型，同时生成链接对象
        self.client.connect(('localhost', 13452))  # 建立一个链接，连接到本地的6969端口

    def run(self):
        while True:
            # addr = client.accept()
            # print '连接地址：', addr
            data = self.client.recv(4096)  # 接收一个信息，并指定接收的大小 为1024字节
            print(data.hex())  # 输出我接收的信息


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='阀门测程序')
    parser.add_argument('-c', default=False, action='store_true', help='是否是开个客户端', required=False)
    args = parser.parse_args()
    if args.c:
        print("运行客户机")
        virtual_valve = VirtualValve()
        virtual_valve.run()
    else:
        print("运行主机")
        valve_tester = ValveTest()
        valve_tester.run()
