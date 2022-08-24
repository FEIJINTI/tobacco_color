import numpy as np
import torch
import os
import cv2
import json

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


root_dir = os.path.split(__file__)[0]

default_config = {'model_name': 'best.pt',
                  'model_path': os.path.join(root_dir, 'weights/'),
                  'conf_thres': 0.5}

cmd_param_dict = {'RL': ['conf_thres', lambda x: (100.0 - int(x)) / 100.0],
                  'MP': ['model_path', lambda x: str(x)],
                  'MN': ['model_name', lambda x: str(x)]}


class SugarDetect(object):
    def __init__(self, model_path):
        self.device = select_device(device='0' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != "cpu"
        self.model = attempt_load(weights=model_path,
                                  map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(640, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16
        # run once if on GPU
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

    @torch.no_grad()
    def detect(self, img, conf_thres=0.5, return_mask=True):
        half, device, model, stride = self.half, self.device, self.model, self.stride
        iou_thres, classes, agnostic_nms, max_det = 0.45, None, True, 1000
        names, imgsz = self.names, self.imgsz

        im0_shape = img.shape

        # Padded resize
        img = letterbox(img, (imgsz, imgsz), stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Preprocess
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        s, det, boxes = "", pred[0], []
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if return_mask:
            mask = np.zeros((im0_shape[0], im0_shape[1]), dtype=np.uint8)
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_shape).round()
            # Print results
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class
            #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if return_mask:
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(mask, c1, c2, 1, thickness=-1)
                else:
                    for i in range(4):
                        boxes.append((int(xyxy[i])))
        if return_mask:
            return mask
        else:
            return boxes


def read_config(config_file):
    config = default_config
    # get config from file
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            json.dump(config, f)
    else:
        with open(config_file, 'r') as f:
            config = json.load(f)
    return config


def write_config(config_file, config=None):
    if config is None:
        config = default_config
    dir_path, _ = os.path.split(config_file)
    if not os.path.exists(dir_path):
        print(f"Path '{dir_path}' not exist, try to create.")
        os.makedirs(dir_path)
    with open(config_file, 'w') as f:
        json.dump(config, f)
    with open(config['model_path']+"current_model.txt", "w") as f:
        f.write(config["model_name"])


def main(height, width, channel):
    img_pipe_path = "/tmp/img_fifo.pipe"
    result_pipe_path = "/tmp/result_fifo.pipe"

    config_file = os.path.join(root_dir, 'config.json')
    config = read_config(config_file)
    detect = SugarDetect(model_path=os.path.join(config['model_path'], config['model_name']))
    # 第一次检测太慢，先预测一张
    test_img = np.zeros((height, width, channel), dtype=np.uint8)
    detect.detect(test_img)
    print("load success")

    if not os.access(img_pipe_path, os.F_OK):  # 判断管道是否存在，不存在创建
        os.mkfifo(img_pipe_path)
    if not os.access(result_pipe_path, os.F_OK):
        os.mkfifo(result_pipe_path)
    fd_img = os.open(img_pipe_path, os.O_RDONLY)  # 打开管道
    print("Open pipe successful.")
    while True:
        data = os.read(fd_img, height * width * channel)
        if len(data) == 0:
            continue
        elif len(data) < 128:  # 切换分选糖果类型
            cmd = data.decode()
            print("to python: ", cmd)
            for cmd_pattern, para_f in cmd_param_dict.items():
                if cmd.startswith(cmd_pattern):
                    para, f = para_f
                    print(f"modify para {para}")
                    try:
                        cmd_value = cmd.split(':')[-1]  # split to get command value with ':'
                        config[para] = f(cmd_value)  # convert value with function defined on the top
                    except Exception as e:
                        print(f"Convert command Error with '{e}'.")
            write_config(config_file, config)
            detect = SugarDetect(model_path=config['model_path']+config['model_name'])
        else:  # 检测缺陷糖果
            img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, channel))
            points = detect.detect(img, config['conf_thres'])

            points_bytes = b''
            if len(points) == 0:
                for i in range(4):
                    points.append(0)
            for i in points:
                points_bytes = points_bytes + i.to_bytes(2, 'big')  # 转为字节流
            fd_result = os.open(result_pipe_path, os.O_WRONLY)
            os.write(fd_result, points_bytes)  # 返回结果
            os.close(fd_result)


if __name__ == '__main__':
    main(height=584, width=2376, channel=3)
