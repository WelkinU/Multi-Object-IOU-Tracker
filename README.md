# Multi-Object IOU Tracker

Re-implementing a Multi-Object Tracker that stitches object bounding boxes together using an intersection-over-union (IOU) metric (via the Hungarian Algorithm).

Advantages of this simple tracking approach:
- Hyper fast runtime
- Should be pretty easy to customize this code
- Minimal parameter tuning required
- No domain knowledge, or dynamics information required

Disadvantages:
- Requires a good object detector, as the IOU tracker doesn't modify the boxes, just stitches
- Requires a fast frame rate to stitch fast-moving small objects
- Can switch tracks on overlapping objects (a Kalman filter implementation in the `get_next_predicted_box()` function in the `Track` class should handle this though)

---

## Features Implemented
- [x] Basic Multi-Object IOU Tracker classes (`MultiObjectTracker`, `Track`, `Box`)
- [x] Basic Test Cases (working on adding more)

## TODO
- [ ] Build track interpolation feature
- [ ] Build easy CSV export
- [ ] Add usage and algorithm documentation to `README.md`
- [ ] Interoperability script with [YOLOv5](https://github.com/ultralytics/yolov5)
- [ ] Kalman filter option added to the `Track` class's `get_next_predicted_box()` function
