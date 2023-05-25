import argparse
import time
from pathlib import Path
# from measure_object_size import measure
import cv2
from object_detector import *
import numpy as np
import pandas as pd
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import cv2
from object_detector import *
import numpy as np
import pandas as pd


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# open('a.txt', 'w').close()
# open('a1.txt', 'w').close()
def detect(save_img=False):
    global im0
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    print(model.names)
    model.names = ['Kutu']
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xywh2 = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist() 
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        line2 = (cls, *xywh2, conf) if opt.save_conf else (cls, *xywh2)  # label format
                        print(xywh2)
                        with open('a.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        with open('a1.txt', 'a') as f:
                            f.write(('%g ' * len(line2)).rstrip() % line2 + '\n')    

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # with open('a.txt', 'a') as f:
                        #     f.write(('%g ' * len(xyxy)).rstrip() % xyxy + '\n')

            # Print time (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')
                # print(xyxy)
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    # measure()
                    
                    wlist=[]
                    hlist=[]
                    g=[]

                    File_data = np.loadtxt('a1.txt', dtype=int)
                    if File_data.size < 6:
                           g=np.delete(File_data, 0)
                    else:
                        # k=len(g[:, 0])

                        g=np.delete(File_data, 0,1)
                    print(File_data)

                    # g=np.delete(File_data, 0,1)
                    y=g.size
                    print(g)
                    # print(len(g[0]))
                    if y < 6:
                           g=g.reshape(1,4)
                    else:
                        # k=len(g[:, 0])

                        g= g.reshape(int((y/4)),4)
                           
                    # x= x.reshape(len(x[:, 0]-1),4)
                    print(g)
                   
                    #  Aruco detektor yükle
                    parameters = cv2.aruco.DetectorParameters_create()
                    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
                    
                    
                    # Nesne detektor yükle
                    detector = HomogeneousBgDetector()
                    
                    # Resim yükle
                    # img = cv2.imread("phone_aruco_marker.jpg")
                    # img = i
                    
                    #  Aruco işaretleyici yükle
                    corners, _, _ = cv2.aruco.detectMarkers(im0, aruco_dict, parameters=parameters)
                    
                    # İşaretçinin etrafına çokgen çiz
                    int_corners = np.int0(corners)
                    cv2.polylines(im0, int_corners, True, (0, 255, 0), 5)
                    
                    # Aruco Çevre uzunluğu
                    aruco_perimeter = cv2.arcLength(corners[0], True)
                    
                    # Piksel cm oranı
                    pixel_cm_ratio = aruco_perimeter / 20
                        
                    contours = detector.detect_objects(im0)
                    if y <5:
                       x = int(g[0,0])
                       y = int(g[0,1])
                       w = int(g[0,2])
                       h = int(g[0,3])
                       rect = ((x, y), (w, h), 90)
                       
                       # Oran pikselini cm'ye uygulayarak Nesnelerin Genişliğini ve Yüksekliğini Al
                       object_width = w / pixel_cm_ratio
                       object_height = h / pixel_cm_ratio
                       wlist.append(object_width)
                       hlist.append(object_height)
                   
                       # dörtgeni göster
                       box = cv2.boxPoints(rect)
                       box = np.int0(box)
                   
                       cv2.circle(im0, (int(x), int(y)), 5, (0, 0, 255), -1)
                       # cv2.polylines(im0, [box], True, (255, 0, 0), 2)
                       # cv2.rectangle(im0, x,y, (255, 0, 0), 2)
                       
                       
                       cv2.putText(im0, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                       cv2.putText(im0, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                       print(x)
                       c=zip(wlist,hlist)
                       # print(list(c))
                       d=list(c)
                       print(d)
                       print(type(d))
                       pd.DataFrame(d).to_excel('output2.xlsx', header=False, index=False)
                    

                    else:
                           v=len(g[:, 0])
                           for i in range(v):
                               
                               # for j in range(3):
                                   # b[i,0] = a[i,0]
                                   # b[i,1] = a[i,1]
                                   # b[i,2] = a[i,1] - a[i,0]
                                   # b[i,3] = a[i,3] - a[i,2]
                                   # 
                               # rect = cv2.minAreaRect(cnt)
                                   x = int(g[i,0])
                                   y = int(g[i,1])
                                   w = int(g[i,2])
                                   h = int(g[i,3])
                                   rect = ((x, y), (w, h), 90)
                                   
                                   # Oran pikselini cm'ye uygulayarak Nesnelerin Genişliğini ve Yüksekliğini Al
                                   object_width = w / pixel_cm_ratio
                                   object_height = h / pixel_cm_ratio
                                   wlist.append(object_width)
                                   hlist.append(object_height)
                               
                                   # dörtgeni göster
                                   box = cv2.boxPoints(rect)
                                   box = np.int0(box)
                               
                                   cv2.circle(im0, (int(x), int(y)), 5, (0, 0, 255), -1)
                                   # cv2.polylines(im0, [box], True, (255, 0, 0), 2)
                                   cv2.putText(im0, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                                   cv2.putText(im0, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                                   # w ve h değerlerini tek tek alıp, birleştirip excel'e koy.
                                   c=zip(wlist,hlist)
                                   # print(list(c))
                                   d=list(c)
                                   print(d)
                                   print(type(d))
                                   pd.DataFrame(d).to_excel('output2.xlsx', header=False, index=False)
                               



                    
                    
                    
                    
                    
              
       
                
                    cv2.imshow("Image", im0)
                    cv2.waitKey(0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
