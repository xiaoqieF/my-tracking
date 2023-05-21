import sys
import torch
import torch.backends.cudnn as cudnn
import time
import cv2
import os
import numpy as np
sys.path.insert(0, './my_centernet')

from botsort.bot_sort import BoTSORT
from deep_sort_pytorch.utils.yaml_parser import get_yaml_data
from my_centernet.networks.centernetplus import CenterNetPlus
from my_centernet.utils.dataset import LoadVideo
from my_centernet.utils.utils import time_sync, scale_boxes, load_class_names
from my_centernet.utils.boxes import postprocess, BBoxDecoder

class_name_path = './my_centernet/my_data_label.names'
bytetrack_config = './botsort/configs/botsort.yaml'
video_path = './5.mp4'
half = True
save_txt = False
txt_path = f'./run/{video_path.split("/")[-1].split(".")[0]}_bot.txt'
show_vid = False
write_video = False

fourcc = cv2.VideoWriter_fourcc(*'mp4v')


if not os.path.exists(os.path.dirname(txt_path)):
    os.makedirs(os.path.dirname(txt_path))

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)
    return img

def detect():
    cfg = get_yaml_data(bytetrack_config)['botsort']
    print(cfg)
    botsort = BoTSORT()
    device = torch.device('cuda:0')

    label_names = load_class_names(class_name_path)

    model = CenterNetPlus(num_classes=1, backbone="r18", pretrained=False)
    model.load_state_dict(torch.load('my_centernet/model_data/DroneVsBirds_centernetplus_r18_best.pth'))
    model.to(device)
    if half:
        model.half()
    model.eval()

    cudnn.benchmark = True  # 加速固定大小图片输入的网络运行
    dataset = LoadVideo(video_path)

    out = cv2.VideoWriter(txt_path.replace('txt', 'mp4'), fourcc, 30, dataset.video_size())

    # gpu 先执行一次 inference
    model(torch.zeros(1, 3, 512, 512).to(device).type_as(next(model.parameters())))

    fpss = []

    with torch.no_grad():
        for frame_idx, (path, img, img0s) in enumerate(dataset):
            start = time.time()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            t1 = time_sync()
            output = model(img)
            output = BBoxDecoder.decode_bbox(output[0], output[1], output[2], confidence=0.2)
            output = postprocess(output, classes=[0])

            t2 = time_sync()

            im0 = img0s.copy()

            for i, det in enumerate(output):
                s = '%gx%g ' % img.shape[2:]
                if len(det) != 0:
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], img0s.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += '%g %ss, ' % (n, label_names[int(c)])

                    outputs = botsort.update(det, im0)
                    outputs = np.array(outputs)

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, 4]
                        draw_boxes(im0, bbox_xyxy, identities)
                        # to MOT format
                        tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                        # Write MOT compliant results to file
                        if save_txt:
                            for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                                bbox_top = tlwh_bbox[0]
                                bbox_left = tlwh_bbox[1]
                                bbox_w = tlwh_bbox[2]
                                bbox_h = tlwh_bbox[3]
                                identity = output[4]
                                with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                                bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                # Print time (inference + NMS)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
            if show_vid:
                cv2.imshow(path, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            if write_video:
                out.write(im0)  
            end = time.time()
            print(f"fps:{1 / (end - start)}")
            fpss.append(1 / (end - start))
        print(f'average fps: {sum(fpss) / len(fpss)}')
        out.release()

if __name__ == '__main__':
    detect()