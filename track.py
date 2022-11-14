import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import math
import psutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'ocsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from utils.augmentations import letterbox
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker
import pandas as pd

import easyocr
# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


### for OCR
reader = easyocr.Reader(["en"],gpu = True)

### for logging
logger_filename = 'detection_results.csv'
logger_data =  pd.DataFrame(columns=['id', 'Class', 'Licence Plate', 'speed'])

################ speed calculation  #############################################
id_centers = {}
def update_speed(outputs, frame_gap = 30):
    global id_centers
    speeds = {}
    centers = np.stack([(outputs[:,2]+outputs[:,0])/2, (outputs[:,3]+outputs[:,1])/2], axis = 1)  
    new_id_centers = {i:cen for i,cen in zip(outputs[:,4], centers)}
    
    if not bool(id_centers):
        id_centers = new_id_centers.copy()
        speeds = {id:0 for id in id_centers.keys()}
        return speeds
    
    all_keys = list(id_centers.keys()) + list(new_id_centers.keys())
    for id in set(all_keys):
        old, new = id_centers.get(id,[0,0]), new_id_centers.get(id,[0,0])
        # print(old)
        # print(new)
        if len(old)!=0 and len(new)!=0 :
            speeds[id]=math.pow(((new[1]-old[1])**2 + (new[0]-old[0])**2), 0.5)/frame_gap
        id_centers[id] = new
    
    return speeds
###################################################################################

label_names = {2: 'car', 5: 'bus', 7: 'truck'}

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        yolo_weights_licence = WEIGHTS / 'yolov5n_license_plate.pt',
        appearance_descriptor_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        stframe=None,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        eval=False,  # run multi-gpu eval
):

    
    global logger_data
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    # exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    if eval:
        device = torch.device(int(device))
    else:
        device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # licence plate
    l_model = DetectMultiBackend(yolo_weights_licence, device=device, dnn=dnn, data=None, fp16=half)
    # stride, names, pt = l_model.stride, l_model.names, l_model.pt
    l_imgsz = check_img_size(640, s=l_model.stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, appearance_descriptor_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        
        im_backup = im.copy()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            #if cfg.STRONGSORT.ECC:  # camera motion compensation
            #    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
            p = 'output'

            if det is not None and len(det):
                
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                
                ##################################################################
                ##   get licence plates for each detection
                
                licence_plates = []
                for k,d in enumerate(det):

                    im_1_backup = im0[int(d[1]):int(d[3]),int(d[0]):int(d[2])].copy()
                    im_1 = letterbox(im_1_backup, l_imgsz, stride=l_model.stride, auto=l_model.pt)[0]  # padded resize
                    im_1_backup_2 = im_1.copy()
                    im_1 = im_1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im_1 = np.ascontiguousarray(im_1) 
                    
                    im_1 = torch.from_numpy(im_1).to(device)
                    im_1 = im_1.half() if half else im_1.float()  # uint8 to fp16/32
                    im_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if len(im_1.shape) == 3:
                        im_1 = im_1[None]  # expand for batch dim

                    p_l = l_model(im_1, augment=augment, visualize=visualize)
                    # Apply NMS
                    p_l = non_max_suppression(p_l, 0.5, 0.3, [0], agnostic_nms, max_det=2)[0]
                    
                    bounds = None
                    if len(p_l)!=0:
                        try:
                            im_ocr = im_1_backup_2[int(p_l[0][1]):int(p_l[0][3]),int(p_l[0][0]):int(p_l[0][2])].copy()
                            bounds = reader.readtext(im_ocr)
                        except:
                            bounds = None
                        p_l[:, :4] = scale_coords(im_1.shape[2:], p_l[:, :4], im_1_backup.shape).round()
                        # cv2.imwrite(f"{k}.jpg", im_1_backup[int(p[0][1]):int(p[0][3]),int(p[0][0]):int(p[0][2])].copy())
                        
                        p_l[:, 0] += d[0]
                        p_l[:, 2] += d[0]
                        p_l[:, 1] += d[1]
                        p_l[:, 3] += d[1]
                    
                    try:
                         if len(bounds[0][1]) >7:
                             lp = bounds[0][1]
                    except:
                        lp = 'None'
                    licence_plates.append((p_l, lp))
                
                ############################################################
                # print("lens :",len(det)==len(licence_plates))
                # print(det)
                # print(licence_plates)
                
                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                ####################################################################################################
                #get speeds
                if seen%30 and len(outputs[i])>0: speeds = update_speed(np.asarray(outputs[i]), frame_gap = 1)
                ####################################################################################################
            
                t5 = time_sync()
                
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf, lp_set) in enumerate(zip(outputs[i], det[:, 4], licence_plates)):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {speeds.get(id,0):.2f}'))
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        
                        LOGGER.info(f'{id} {names[c]} {speeds.get(id,0):.2f}')
                        
                        # show licence plate
                        lp_bb, lp = lp_set
                        if len(lp_bb)!=0:
                            annotator.box_label(lp_bb[0][0:4], lp, color=colors(c, True))
                        
                        if speeds.get(id,0) > 50:
                            sp = 50
                        else:
                            sp = speeds.get(id,0)
                        
                        
                        if id in logger_data['id']:
                            
                            
                            index = logger_data.loc[logger_data['id']==id].index[0]
                            entry = logger_data.iloc[index]
                            if lp == 'None':
                                logger_data.iloc[index] = [id, label_names[c], entry['Licence Plate'], sp]
                            else:
                                logger_data.iloc[index] = [id, label_names[c], lp, sp]
                            
                        else:
                            logger_data = logger_data.append({'id' : id, 'Class':label_names[c] , 
                                                              'Licence Plate':lp, 'speed': sp}, ignore_index=True)
                        
                
                LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')

            else:
                #strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            # if show_vid:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            stframe.image(im0, channels="BGR",use_column_width=True)
            
            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
        
            
    # Create the pandas DataFrame
    # df = pd.DataFrame(logger_data, columns = ['id', 'Class', 'Licence Plate', 'speed'])
    # print(df)
    # df.to_csv(logger_filename, index = False)
    # idx = np.unique(df[['id', 'Licence Plate']].values, return_index=1)[-1]
    # print(idx)
    # df = df.filter(items=idx, axis=0)
    logger_data.reset_index(drop = True, inplace = True)
    logger_data.to_csv(logger_filename, index = False)
    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--yolo-weights-licence', nargs='+', type=Path, default=WEIGHTS / 'yolov5n_license_plate.pt', help='model.pt path(s)')
    parser.add_argument('--appearance-descriptor-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='video_test.mp4', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car, 5 is bus, 7 is truck... 79 is oven
    parser.add_argument('--classes', nargs='+', default=[2,5,7], type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--stframe',default =None)

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)