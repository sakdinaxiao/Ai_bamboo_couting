from clahe_inference import apply_clahe
from ultralytics import YOLO
from sahi_inference import apply_sahi,get_sahi
from pathlib import Path
import cv2
import supervision as sv
from bytetrack_inference import get_bytetrack, get_counting_zone, tracking
from segmentation import segmenting
import numpy as np
import argparse

project_root = Path(__file__).resolve().parent
model_path = project_root / "training_result" / "detection_small" / "weights" / "best.pt" 
seg_model_path = project_root / "training_result" / "segment" / "best.pt"

#using small YOLO

def main(video):
    if not seg_model_path.exists():
        print("Segmentation model path does not exist")
        return
    seg_model = YOLO(seg_model_path)
    if not model_path.exists():
        print("Model path does not exist")
        return

    source_vid = project_root / video
    if not source_vid.exists():
        print("There no video")
        return

    cap = cv2.VideoCapture(str(source_vid))
    if not cap.isOpened():
        print("Can't open video")
        return
    
    try:

        sahi = get_sahi(model_path)
        
        bytetrack, id_counter = get_bytetrack()
        min_x, min_y, max_x, max_y = get_counting_zone((frame_height, frame_width, 3))
        
        #----- for visual only
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        trap_points = np.array([
            [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]
        ], dtype=np.int32)

        zone = sv.PolygonZone(polygon=trap_points) 
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color(255,0,0), thickness=8)
        #-----

        origin_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = 3
        stride = max(1, int(origin_fps/target_fps))

        frame_count = 0

        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            
            frame_count += 1

            if frame_count % stride != 0:
                continue
            
            regions = segmenting(seg_model,frame)
            all_detections_xyxy = []
            all_confidence = []
            all_class_ids=[]

            for region in regions:
                reg_img = region["image"]
                offset_x = region["offset_x"]
                offset_y = region["offset_y"]

                img = apply_clahe(frame=reg_img)
                detected = apply_sahi(sahi,img)

                if not detected.is_empty():
                    for i in range(len(detected.xyxy)):
                        remaped=[
                            detected.xyxy[i][0] + offset_x,
                            detected.xyxy[i][1] + offset_y,
                            detected.xyxy[i][2] + offset_x,
                            detected.xyxy[i][3] + offset_y 
                        ]
                        all_detections_xyxy.append(remaped)
                        all_confidence.append(detected.confidence[i])
                        all_class_ids.append(detected.class_id[i])

            if len(all_detections_xyxy) > 0:
                final_detections = sv.Detections(
                    xyxy=np.array(all_detections_xyxy),
                    confidence=np.array(all_confidence),
                    class_id=np.array(all_class_ids)
                )
            else:
                final_detections = sv.Detections.empty()

            print(f"Raw YOLO Detections this frame: {len(all_detections_xyxy)}")

            tracked_detections = tracking(
                final_detections,
                bytetrack,
                id_counter,
                (min_x, min_y, max_x, max_y),
            )

            # -------------------------- Visulization
            labels = [
                f"#{tracker_id}"
                for tracker_id in tracked_detections.tracker_id
            ] if tracked_detections.tracker_id is not None else []

            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
            annotated_frame = zone_annotator.annotate(scene=annotated_frame)
            
            cv2.putText(
                annotated_frame,
                f"Total Counted: {len(id_counter)}",
                (40, 70), 
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0, 
                (0, 255, 0), 
                4 
            )
            
            cv2.imshow("Tracking & Counting Debugger", annotated_frame)

            if cv2.waitKey(1) == ord('q'):
                break
            
            # ---------------------------
        
        return len(id_counter)
    
    except Exception as e:
        print(f"Error {e}")
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bamboo counter")
    parser.add_argument("--source", type=str)
    args = parser.parse_args()
    print(main(args.source))
    
