''' Multi-Object IOU Tracker

A simple online multi-object tracker that stitches together bounding boxes via 
IOU overlap. Designed as a very fast tracker to work with YOLO algorithms.

Advantages over more complicated trackers (such as those in OpenCV):
- Computationally very fast
- Minimal parameter tuning
- No domain knowledge needed
'''

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

class MultiObjectTracker:

    def __init__(self, track_persistance: int = 1, 
                minimum_track_length: int = 1,
                iou_lower_threshold: float = 0.04,
                interpolate_tracks: bool = False,
                cross_class_tracking: bool = True):

        assert track_persistance >= 0
        assert 0 <= iou_lower_threshold <= 1
        assert minimum_track_length > 0

        self.track_persistance = track_persistance
        self.iou_lower_threshold = iou_lower_threshold
        self.minimum_track_length = minimum_track_length
        self.interpolate_tracks = interpolate_tracks
        self.cross_class_tracking = cross_class_tracking

        self.active_tracks = []
        self.finished_tracks = []

        self.time_step = 0

    def step(self,list_of_new_boxes: list = []):
        ''' Call this method to add more bounding boxes to the tracker'''

        self.time_step += 1 #increment time step

        #build hungarian matrix of bbox IOU's
        hungarian_matrix = np.zeros((len(self.active_tracks), len(list_of_new_boxes)))

        if len(self.active_tracks) > 0 and len(list_of_new_boxes) > 0:
            active_boxes = np.concatenate([track.get_next_predicted_box()[np.newaxis,:]
                            for track in self.active_tracks], axis = 0)

            new_boxes = np.concatenate([ b['box'][np.newaxis,:] for b in list_of_new_boxes], 
                                        axis = 0)
            hungarian_matrix = IOU(new_boxes,active_boxes)

            #print(hungarian_matrix)

            #zero out IOU's less than IOU min threshold to prevent assigment
            hungarian_matrix[hungarian_matrix < self.iou_lower_threshold] = 0

            #zero out IOU's where the object_class variables don't match
            if not self.cross_class_tracking:
                for i,new_box in enumerate(list_of_new_boxes):
                    if "object_class" in new_box:
                        for j,active_track in enumerate(self.active_tracks):
                            active_box = active_track.boxes[0] #shouldn't matter which box in the track
                            if "object_class" in active_box and \
                                are_coco_classes_different(new_box['object_class'], active_box['object_class']):
                                hungarian_matrix[i,j] = 0

            #print(hungarian_matrix)

            #compute optimal box matches with Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(hungarian_matrix, maximize = True)

            #assign new boxes to active tracks, boxes not assigned this way become new tracks
            for i,box in enumerate(list_of_new_boxes):
                if i in row_ind:
                    #if the new box has matching active track, add it to that track
                    r = (row_ind==i).argmax(axis=0)
                    c = col_ind[r]
                    if hungarian_matrix[r,c] > 0:
                        self.active_tracks[c].add_box(box, self.time_step)
                    else:
                        #new box has no matching active track, create a new track for it
                        self.active_tracks.append(Track(initial_box = box, initial_time_step = self.time_step))
                else:
                    #new box has no matching active track, create a new track for it (same as row above)
                    self.active_tracks.append(Track(initial_box = box, initial_time_step = self.time_step))

        else:
            for box in list_of_new_boxes:
                self.active_tracks.append(Track(initial_box = box, initial_time_step = self.time_step))

        #active tracks with age >= track_persistance are set to be finished tracks
        newly_finished_track_ind = []
        for i, trk in enumerate(self.active_tracks):
            if trk.timestamps[-1] + self.track_persistance < self.time_step:
                self.finished_tracks.append(trk)
                newly_finished_track_ind.append(i)

        self.active_tracks = [element for i,element in enumerate(self.active_tracks) if i not in newly_finished_track_ind]

    def finish_tracking(self):
        ''' Call this method when you are done adding new boxes.

        Finish all active tracks, do interpolation if selected, prune tracks shorter than 
        minimum_track_length parameter '''
        self.finished_tracks.extend(self.active_tracks)
        self.active_tracks = []

        if self.interpolate_tracks:
            for i in range(len(self.finished_tracks)):
                self.finished_tracks[i].interpolate()

        #prune tracks with length less than minimum_track_length
        self.finished_tracks = [trk for trk in self.finished_tracks if len(trk) >= self.minimum_track_length]

        #sort tracks by track start time
        self.finished_tracks = sorted(self.finished_tracks, key = lambda x: x.timestamps[0])

    def export_pandas_dataframe(self, additional_cols = 'auto'):
        '''Converts multi-object tracker internal state to Pandas dataframe with cols:
        Time, TrackID, X1, Y1, X2, Y2

        Arguments:
        additional_cols {list or str} -- List of additional attributes to grab from the Box class. 
                                For each attribute, a column will be added to dataframe and
                                the function will attempt to grab that value from each Box object
                                added to the dataframe. If the attribute is not present in a
                                Box object, it will add NaN or None.

                                If this variable is set to the string "auto", it will populate this
                                variable with all extra params in all Box added to the tracker. 

        Returns:
        Pandas DataFrame with specified rows'''

        if isinstance(additional_cols,str) and additional_cols in ['Auto','auto']:
            additional_cols = set()

            for trk in self.finished_tracks:
                for box in trk.boxes:
                    additional_cols = additional_cols.union(set(box.keys()))

            additional_cols.discard("box")
            additional_cols = list(additional_cols)

        df_list = []

        for time in range(self.time_step + 1):
            for track_id, trk in enumerate(self.finished_tracks):
                if time in trk.timestamps:
                    box = trk.boxes[trk.timestamps.index(time)]
                    row = {'Time': time,
                            'TrackID': track_id,
                            'X1': box['box'][0],
                            'Y1': box['box'][1],
                            'X2': box['box'][2],
                            'Y2': box['box'][3],
                            }
                    for col in additional_cols:
                        row[col] = box[col] if col in box.keys() else None

                    df_list.append(row)

        return pd.DataFrame(df_list)

    def print_internal_state(self):
        ''' For debugging purposes'''

        print('###########################################')
        print(f'Time Step: {self.time_step}')
        print('---------------Active Tracks---------------')
        for i,trk in enumerate(self.active_tracks):
            print(f'Track {i}: { [list(b["box"]) for b in trk.boxes] }')

        print('--------------Finished Tracks--------------')
        for i,trk in enumerate(self.finished_tracks):
            print(f'Track {i}: {[list(b["box"]) for b in trk.boxes]}')
        print('###########################################')


class Track:

    def __init__(self, initial_box, initial_time_step):
        self.boxes = [initial_box]
        self.timestamps = [initial_time_step]

    def get_next_predicted_box(self):
        #maybe add a kalman filter here or something

        return self.boxes[-1]['box']

    def add_box(self, box, time_step):
        self.boxes.append(box)
        self.timestamps.append(time_step)

    def interpolate(self):
        ''' TODO: Implement this function'''
        interpolated_boxes = []

        for i in range(len(self.boxes) - 1):
            interpolated_boxes.append(self.boxes[i])

            delta = self.timestamps[i+1] - self.timestamps[i]
            if delta == 1:
                continue #no need to interpolate sequential boxes

            #not interpolating confidence or other numerical parameters
            new_boxes = [{"box": ((delta - j) * self.boxes[i]['box'] + j * self.boxes[i+1]['box'])/delta}
                            for j in range(1, delta)]

            interpolated_boxes.extend(new_boxes)

        interpolated_boxes.append(self.boxes[-1])
        self.boxes = interpolated_boxes
        self.timestamps = list(range(self.timestamps[0], self.timestamps[-1] + 1))

        assert len(self.boxes) == len(self.timestamps), f'length of boxes ({len(self.boxes)} != length timestamps ({len(self.timestamps)})'


    def __len__(self):
        return self.timestamps[-1] - self.timestamps[0] + 1

def IOU(bboxes1, bboxes2, isPixelCoord = 1):
    #vectorized IOU numpy code from:
    #https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d

    #input N x 4 numpy arrays
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + isPixelCoord), 0) * np.maximum((yB - yA + isPixelCoord), 0)
    boxAArea = (x12 - x11 + isPixelCoord) * (y12 - y11 + isPixelCoord)
    boxBArea = (x22 - x21 + isPixelCoord) * (y22 - y21 + isPixelCoord)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def are_coco_classes_different(c1, c2):
    ''' Sets which classes are "equivalent" for the tracker. Most classes are not equal
    but objects like pickup trucks can be detected as both car and truck. Or sometimes
    busses can be detected as both truck and bus. This function is a quick and dirty
    way to stop the track from breaking.
    '''
    return c1 != c2 and ( {c1,c2} not in [{'car','truck'}, {'bus','truck'}])


##############################################
###-------------Testing Zone---------------###
##############################################

if __name__ == '__main__':
    
    mot = MultiObjectTracker(track_persistance = 3, minimum_track_length = 2, iou_lower_threshold = 0.04, interpolate_tracks = True)

    #add some initial boxes
    boxes = [{"box": np.array([0,0,2,2]), "confidence": 0.9, "object_class": "car"}, 
            {"box": np.array([10,10,12,12]), "confidence": 0.8, "object_class": "truck"}]
    mot.step(boxes)
    mot.print_internal_state()

    #add same boxes, plus a new box
    boxes = [{"box": np.array([0,0,2,2]), "confidence": 0.7, "object_class": "car"}, 
            {"box":np.array([10,10,12,12])},
            {"box":np.array([20,20,22,22])}]
    mot.step(boxes)
    mot.print_internal_state()

    #add same first 2 boxes, then change the new box, overlapping with first
    boxes = [{"box":np.array([0,0,1,1]), "object_class": "car" },
            {"box":np.array([0,0,2,2]), "object_class": "notacar"},
            {"box":np.array([10,10,12,12])}]
    mot.step(boxes)
    mot.print_internal_state()

    boxes = [{"box":np.array([0,0,1,1]), "object_class": "car" }]
    mot.step(boxes)
    mot.print_internal_state()
    boxes = []
    mot.step(boxes)
    mot.step(boxes)
    mot.print_internal_state()
    boxes = [{"box":np.array([0,0,0.5,0.5]), "object_class": "truck" },
             {"box":np.array([0,0,2,2]), "object_class": "car"}]
    mot.step(boxes)
    mot.print_internal_state()
    boxes = [{"box":np.array([0,0,1,1]), "object_class": "notacar" },
             {"box":np.array([0,0,2,2]), "object_class": "car"}]
    mot.step(boxes)
    mot.print_internal_state()
    boxes = [{"box":np.array([0,0,1,1]), "object_class": "truck" },
             {"box":np.array([0,0,2,2]), "object_class": "yolo"}]
    mot.step(boxes)
    mot.print_internal_state()

    #add some steps with no boxes, tracks move to finished
    mot.step([])
    mot.print_internal_state()
    mot.step([])
    mot.print_internal_state()

    #finish tracking 
    mot.finish_tracking()
    mot.print_internal_state()

    df = mot.export_pandas_dataframe(additional_cols = 'auto')
    print(df)
    #df.to_csv('test.csv', index=False)
