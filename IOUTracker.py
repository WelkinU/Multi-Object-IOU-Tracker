''' IOU Bounding Box Multi-Object Tracker
'''

import numpy as np
from scipy.optimize import linear_sum_assignment

class MultiObjectTracker:

    def __init__(self, track_persistance = 3, iou_lower_threshold = 0.04, minimum_track_length = 4):
        self.track_persistance = track_persistance
        self.iou_lower_threshold = iou_lower_threshold
        self.minimum_track_length = minimum_track_length

        self.active_tracks = []
        self.finished_tracks = []

        self.time_step = 0

    def step(self,list_of_new_boxes):

        self.time_step += 1 #increment time step

        #build hungarian matrix of bbox IOU's
        hungarian_matrix = np.zeros((len(self.active_tracks), len(list_of_new_boxes)))
        
        '''
        for i, active_box in enumerate(self.active_tracks):
            for j, new_box in enumerate(list_of_new_boxes):
                hungarian_matrix[i,j] = IOU(active_box,new_box)
        '''
        active_boxes = np.vstack([track.get_next_predicted_box().box for track in self.active_tracks])
        new_boxes = np.vstack(list_of_new_boxes)
        hungarian_matrix = IOU(new_boxes,active_boxes)

        #compute optimal box matches with Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(hungarian_matrix, maximize = True)

        #assign new boxes to active tracks, boxes not assigned this way become new tracks
        for i,box in enumerate(list_of_new_boxes):
            if i in row_ind:
                #if the new box has matching active track, add it to that track
                self.active_tracks[col_ind[row_ind.index(i)]].add_box(box, self.time_step)
            else:
                #new box has no matching active track, create a new track for it
                self.active_tracks.append(Track(initial_box = box, initial_time_step = self.time_step))

        #active tracks with age > track_persistance are set to be finished tracks
        newly_finished_track_ind = []
        for i, trk in enumerate(self.active_tracks):
            if trk.last_added_time_step + track_persistance < self.time_step:
                self.finished_tracks.append(trk)
                newly_finished_track_ind.append(i)

        self.active_tracks = [element for i,element in enumerate(self.active_tracks) if i not in newly_finished_track_ind]


    def finish_tracking(self):
        '''Finish all active tracks, do interpolation if selected, prune tracks shorter than 
        minimum_track_length parameter '''
        pass

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

class Box:

    def __init__(self, box = np.array([0,0,0,0]), **kwargs):
        '''

        Arguments:
        box {1x4 numpy array} -- X1, Y1, X2, Y2
        '''

        self.box = box 
        self.box_properties = kwargs

    def __str__(self):
        return f'''Box: {self.box}, Properties: {self.box_properties}'''



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
    pass
