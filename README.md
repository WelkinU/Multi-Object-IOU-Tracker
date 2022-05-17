# Multi-Object IOU Tracker

Implementing a Multi-Object Tracker that stitches object bounding boxes together using an intersection-over-union (IOU) metric (via the Hungarian Algorithm).

Advantages of this simple tracking approach:
- Hyper fast runtime
- Online tracker (doesn't need future frames to do tracking)
- Easy to customize
- Minimal parameter tuning required
- No domain knowledge, or dynamics information required

Disadvantages:
- Requires a good object detector, as the IOU tracker doesn't modify the boxes, just stitches
- Requires a fast frame rate to stitch fast-moving small objects
- Can switch tracks on overlapping objects (a Kalman filter implementation in the `get_next_predicted_box()` function in the `Track` class should handle this though)

---

## Example Usage

``` Python
from IOUTracker import MultiObjectTracker
mot = MultiObjectTracker() #initialize MultiObjectTracker object

# Example of how to add bounding boxes at each time step
# You'd probably do this with a For loop in actual code

#Timestep #1: add some boxes, boxes are dictionaries with the key "box" set to a
#numpy array containing X1,Y1,X2,Y2
boxes = [{"box": np.array([0,0,2,2]) }, 
        {"box": np.array([10,10,12,12]) } ]
mot.step(boxes)

#Timestep #2: add the same boxes, plus a new box
boxes = [{"box": np.array([0,0,2,2]) }, 
        {"box":np.array([10,10,12,12]) },
        {"box":np.array([20,20,22,22]) } ]
mot.step(boxes)

#Timestep #3: You can attach any args to a box
boxes = [ {"box": np.array([0,0,2,2]), "confidence": 0.9, "object_class": "car"} ]
mot.step(boxes)

# Finish tracking - call this function when you're done adding bounding boxes
mot.finish_tracking()

# Export state as Pandas DataFrame
# cols: Time, TrackID, X1, Y1, X2, Y2 + whatever additional args that were in the dicts above
df = mot.export_pandas_dataframe()

df.to_csv('output.csv', index = False) #export to CSV
```

## MultiObjectTracker Class Optional Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| track_persistance | 1 | If a Track hasn't had a bounding box added in more than `track_persistance` time steps, then the track is ended. |
| minimum_track_length | 1 | After tracking is complete, tracks with length less than `minimum_track_length` are deleted |
| iou_lower_threshold | 0.04 | A bounding box needs a minumum IOU of `iou_lower_threshold` to be added to an existing track |
| interpolate_tracks | False | After tracking is complete, tracks with "missing boxes" are interpolated (linear interpolation) **Not Yet Implemented** |
| box_prediction_method | None | Method used to predict the next box location for the track. [`None`, `Linear`, `Kalman Filter`] **Not Yet Implemented** |

Example usage: `mot = MultiObjectTracker(track_persistance = 2, minimum_track_length = 4)`

---

## Algorithm Description

1. Initialize Multi-Object Tracker with empty lists for Active and Finished Tracks
1. Feed in the list of boxes for next time step.
1. Compute the IOU for each new box against each Active Track's `next_predicted_box()` 
1. Use the Hungarian Algorithm to find the optimal assignment of new boxes to Active Tracks.
1. For each new box, assign it to it's associated Active Track. For boxes not associated to an Active Track, start a new Active Track with that box as the seed.
1. For each Active Track, if a box hasn't been added to the track in > `track_persistance` time steps, move that track to be a Finished Track.
1. Repeat steps 2-6 for each time step (until no bounding boxes remain to add/track)
1. Move all Active Tracks to be Finished Tracks.
1. Interpolate each track if that option has been selected by user.
1. Prune tracks shorter than the `minimum_track_length` parameter 

---

## Features Implemented

- [x] Basic Multi-Object IOU Tracker classes (`MultiObjectTracker`, `Track`, `Box`)
- [x] Basic Test Cases (working on adding more)
- [x] Build Pandas Dataframe / CSV export
- [x] Add usage and algorithm documentation to `README.md`
- [x] Interoperability script with [YOLOv5](https://github.com/ultralytics/yolov5)
- [x] Video demo in conjunction with [YOLOv5](https://github.com/ultralytics/yolov5)
- [x] Build track interpolation feature

## Currently In Work

- [ ] Prevent objects of different classes from being added to same track (with some exceptions).

## TODO

- [ ] Kalman filter option added to the `Track` class's `get_next_predicted_box()` function

