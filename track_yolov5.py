
import numpy as np
import pandas as pd
import os
import re
import cv2
import random
from tqdm import tqdm

from IOUTracker import MultiObjectTracker

def track(yolov5_labels_folder):
    mot = MultiObjectTracker(track_persistance = 1, minimum_track_length = 7, iou_lower_threshold = 0.2)

    #get list of natural sorted label files
    label_file_list = [os.path.join(yolov5_labels_folder, file) 
                        for file in natural_sort(os.listdir(yolov5_labels_folder)) 
                        if file[-4:] == '.txt']
    
    for i,label_file in enumerate(tqdm(label_file_list)):
        #use pandas to read the CSV
        df = pd.read_csv(label_file, delimiter = ' ', names = ['object_class','x_center','y_center','width','height','confidence'])

        list_of_boxes = []
        for row in df.itertuples():
            #convert X Y W H to X1 Y1 X2 Y2
            box_to_add = np.array([row.x_center - row.width/2, 
                        row.y_center - row.height/2,
                        row.x_center + row.width/2, 
                        row.y_center + row.height/2])

            list_of_boxes.append({"box": box_to_add,
                                "confidence": row.confidence,
                                "object_class": class_labels[row.object_class]})

        mot.step(list_of_boxes)

        #if i > 400:
        #    break

    print('Finishing tracking...')
    mot.finish_tracking()

    print('Exporting dataframe...')
    df = mot.export_pandas_dataframe(additional_cols = 'auto')

    return df

def generate_video(video_filepath, df, output_video_name):

    cap = cv2.VideoCapture(video_filepath)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_codec = cv2.VideoWriter_fourcc(*"mp4v") #int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'Writing Video: height: {height} width: {width} video_codec: {video_codec} fps: {fps}')
    vid_writer = cv2.VideoWriter(output_video_name, video_codec, fps, (width, height))

    df = df.set_index('Time')
    ret = True
    i = 0
    while ret:
        ret, frame = cap.read()
        i += 1
        
        for row in df.loc[i:i].itertuples():
            box = [int(row.X1 * width),int(row.Y1 * height),int(row.X2 * width),int(row.Y2 * height)]
            #label = f'{class_labels[row.object_class]}{row.TrackID} {row.confidence:.2f}'
            label = f'{row.object_class}{row.TrackID}'
            plot_one_box(box, frame, color = colors[row.object_class], line_thickness = 2,
                        label = label)

        vid_writer.write(frame)

        #if i>400:
        #    break

    cap.release()
    vid_writer.release()
    print('Done.')

def natural_sort(l): 
    #natural sort file list
    #from https://stackoverflow.com/questions/11150239/natural-sorting
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Directly copied from: https://github.com/ultralytics/yolov5/blob/cd540d8625bba8a05329ede3522046ee53eb349d/utils/plots.py
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 5, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    import argparse 
    #default_video_filepath = r"C:\Users\W\Desktop\dev\yolov5\inference\videos\Street Cycling.mp4"
    #default_yolov5_labels_folder = r"C:\Users\W\Desktop\dev\yolov5\runs\detect\exp7\labels"
    default_video_filepath = r"C:\Users\W\Desktop\dev\yolov5\inference\videos\Dashcam.mp4"
    default_yolov5_labels_folder = r"C:\Users\W\Desktop\dev\yolov5\runs\detect\exp8\labels"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-folder', type=str, default=default_yolov5_labels_folder, help='Folder containing YOLOv5 output labels')
    parser.add_argument('--source', type=str, default=default_video_filepath, help='Source Video Filepath')
    parser.add_argument('--out', type=str, default="output.mp4", help='Output video name')
    opt = parser.parse_args()

    #from YOLOv5/data/coco128.yaml
    class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names


    colors = {cl: tuple([random.randint(0, 255) for _ in range(3)]) 
                for cl in class_labels}

    df = track(opt.label_folder)
    #df.to_csv('output.csv', index = False)
    generate_video(opt.source, df, opt.out)