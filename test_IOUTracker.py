from pandas.testing import assert_frame_equal


def one_box_tracking_test():
    ''' Simple test. Inputting 1 step with 1 box then finishing tracking and ensuring result is same as input'''
    mot = MultiObjectTracker()

    #add some initial boxes
    boxes = [{"box": np.array([0,0,2,2]), "confidence": 0.9, "object_class": "car"}, 
            {"box": np.array([10,10,12,12]), "confidence": 0.8, "object_class": "truck"}]

    mot.step(boxes)

    mot.finish_tracking()

    df = mot.export_pandas_dataframe(additional_cols = 'auto')

    expected_result = pd.DataFrame(boxes)
    assert assert_frame_equal(df.sort_index(axis=1), 
            expected_result.sort_index(axis=1), 
            check_names=True)

def basic_functionality():
    ''' Tests basic functionality by adding similar boxes on sequential time steps and verifying they're tracked
    properly. Also minimum_track_length functionality is tested. IOU lower thresh and interpolate not tested here.'''

    mot = MultiObjectTracker(minimum_track_length = 2)

    #add some initial boxes
    boxes = [{"box": np.array([0,0,2,2])}, 
            {"box": np.array([10,10,12,12])}]
    mot.step(boxes)
    mot.print_internal_state()

    #add same boxes, plus a new box
    boxes = [{"box": np.array([0,0,2,2])}, 
            {"box":np.array([10,10,12,12])},
            {"box":np.array([20,20,22,22])}]
    mot.step(boxes)
    mot.print_internal_state()

    #add same first 2 boxes, then change the new box, overlapping with first
    boxes = [{"box":np.array([0,0,1,1])},
            {"box":np.array([0,0,2,2])},
            {"box":np.array([10,10,12,12])}]
    mot.step(boxes)
    mot.print_internal_state()

    mot.finish_tracking()
    df = mot.export_pandas_dataframe(additional_cols = 'auto')

    expected_result = pd.DataFrame({
        'Time': [1,1,2,2,3,3],
        'TrackID': [0,1,0,1,0,1],
        'X1': [0,10,0,10,0,10],
        'Y1': [0,10,0,10,0,10],
        'X2': [2, 12, 2, 12, 2, 12],
        'Y2': [2, 12, 2, 12, 2, 12],
        })

    assert np.array_equal(df.values, expected_result.values)

def interpolate_basic():
    '''Tests basic functionality of the Track class's interpolate() function
    Interpolates between [0,0,2,2] -> [0,0,1,1] over 3 interim time steps'''

    mot = MultiObjectTracker(track_persistance = 4, interpolate_tracks = True)

    boxes = [{"box": np.array([0,0,2,2])}] 
    mot.step(boxes)

    mot.step([])
    mot.step([])
    mot.step([])

    boxes = [{"box": np.array([0,0,1,1])}] 
    mot.step(boxes)

    mot.finish_tracking()
    df = mot.export_pandas_dataframe(additional_cols = 'auto')

    expected_result = pd.DataFrame({
        'Time': [1,2,3,4,5],
        'TrackID': [0,0,0,0,0],
        'X1': [0.,0.,0.,0.,0.],
        'Y1': [0.,0.,0.,0.,0.],
        'X2': [2, 1.75, 1.5, 1.25, 1],
        'Y2': [2, 1.75, 1.5, 1.25, 1],
        })

    assert np.array_equal(df.values, expected_result.values)
    #for some reason this next line doesn't work...
    #assert assert_frame_equal(df, expected_result,check_like = True, check_dtype = False, check_exact = False)

if __name__ == '__main__':
    from IOUTracker import MultiObjectTracker
    import numpy as np
    import pandas as pd
    basic_functionality()