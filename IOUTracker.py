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

    def __init__(self, track_persistance: int = 2, 
                iou_lower_threshold: float = 0.04,
                minimum_track_length: int = 2,
                interpolate_tracks: bool = False):

        assert track_persistance >= 0
        assert 0 <= iou_lower_threshold <= 1
        assert minimum_track_length > 0

        self.track_persistance = track_persistance
        self.iou_lower_threshold = iou_lower_threshold
        self.minimum_track_length = minimum_track_length
        self.interpolate_tracks = interpolate_tracks

        self.active_tracks = []
        self.finished_tracks = []

        self.time_step = 0

    def step(self,list_of_new_boxes = []):
        ''' Call this method to add more bounding boxes to the tracker'''

        self.time_step += 1 #increment time step

        #build hungarian matrix of bbox IOU's
        hungarian_matrix = np.zeros((len(self.active_tracks), len(list_of_new_boxes)))

        if len(self.active_tracks) > 0 and len(list_of_new_boxes) > 0:
            active_boxes = np.concatenate([track.get_next_predicted_box().box[np.newaxis,:]
                            for track in self.active_tracks], axis = 0)
            new_boxes = np.concatenate([b.box[np.newaxis,:] for b in list_of_new_boxes], axis = 0)
            hungarian_matrix = IOU(new_boxes,active_boxes)

            #zero out IOU's less than IOU min threshold to prevent assigment
            hungarian_matrix[hungarian_matrix < self.iou_lower_threshold] = 0

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

        #active tracks with age > track_persistance are set to be finished tracks
        newly_finished_track_ind = []
        for i, trk in enumerate(self.active_tracks):
            if trk.last_added_time_step + self.track_persistance <= self.time_step:
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
                self.finished_tracks.interpolate()

        #prune tracks with length less than minimum_track_length
        self.finished_tracks = [trk for trk in self.finished_tracks if len(trk) >= self.minimum_track_length]

        #sort tracks by track start time
        self.finished_tracks = sorted(self.finished_tracks, key = lambda x: x.initial_time_step)

    def print_internal_state(self):
        ''' For debugging purposes'''

        print('###########################################')
        print(f'Time Step: {self.time_step}')
        print('---------------Active Tracks---------------')
        for i,trk in enumerate(self.active_tracks):
            print(f'Track {i}: {[list(b.box) for b in trk.boxes]}')

        print('--------------Finished Tracks--------------')
        for i,trk in enumerate(self.finished_tracks):
            print(f'Track {i}: {[list(b.box) for b in trk.boxes]}')
        print('###########################################')


class Track:

    def __init__(self, initial_box, initial_time_step):
        self.boxes = [initial_box]
        self.initial_time_step = initial_time_step

        self.last_added_time_step = initial_time_step

    def get_next_predicted_box(self):
        #maybe add a kalman filter here or something

        return self.boxes[-1]

    def add_box(self, box, time_step):
        self.boxes.append(box)
        self.last_added_time_step = time_step

    def interpolate(self):
        pass

    def __len__(self):
        return self.last_added_time_step - self.initial_time_step + 1

class Box:

    def __init__(self, box = np.array([0,0,0,0]), **kwargs):
        '''

        Arguments:
        box {1x4 numpy array} -- X1, Y1, X2, Y2
        '''
        if isinstance(box, list):
            box = np.array(box)

        self.box = box 
        self.box_properties = kwargs

    def __str__(self):
        return f'''Box: {str(self.box)}, Properties: {self.box_properties}'''

def IOU(bboxes1, bboxes2):
    #vectorized IOU numpy code from:
    #https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d

    #input Nx4 numpy arrays
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

##############################################
###-------------Testing Zone---------------###
##############################################

if __name__ == '__main__':
    
    mot = MultiObjectTracker(track_persistance = 2, minimum_track_length = 2)

    #add some initial boxes
    boxes = [Box([0,0,2,2]), Box([10,10,12,12])]
    mot.step(boxes)
    mot.print_internal_state()

    #add same boxes, plus a new box
    boxes = [Box([0,0,2,2]), Box([10,10,12,12]), Box([20,20,22,22])]
    mot.step(boxes)
    mot.print_internal_state()

    #add same first 2 boxes, then change the new box, overlapping with first
    boxes = [Box([0,0,1,1]), Box([0,0,2,2]), Box([10,10,12,12])]
    mot.step(boxes)
    mot.print_internal_state()

    #add some steps with no boxes, tracks move to finished
    mot.step([])
    mot.print_internal_state()
    mot.step([])
    mot.print_internal_state()

    #finish tracking and 
    mot.finish_tracking()
    mot.print_internal_state()