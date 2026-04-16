import supervision as sv
import numpy as np 

def get_bytetrack():
    tracker =  sv.ByteTrack(
            track_activation_threshold=0.4, 
            lost_track_buffer=100, 
            minimum_matching_threshold=0.5,
            minimum_consecutive_frames=1
)
    counter = set()


    return tracker, counter

def tracking(detected,bytetrackmodel,counter):
    track =  bytetrackmodel.update_with_detections(detected)

    if track.tracker_id is not None:
        
        for bbox, tracker_id in zip(track.xyxy, track.tracker_id):
            
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            if (768 < center_x < 3072) and (432 < center_y < 1728):
                counter.add(tracker_id)
    
    return track