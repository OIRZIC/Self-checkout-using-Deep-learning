from ultralytics import YOLO
import numpy as np
import cv2
import database
import datetime
import os

# import time
# import matplotlib.pyplot as plt
# import pandas as pd

import torch
from byte_tracker.tracker.byte_tracker import BYTETracker

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# load yolo model and make it run on GPU instead of CPU
model = YOLO("../weights/best_v8.pt").to(device)


#Index to class
indextoclass = [
    'BUT_CHI_DIXON', 'BUT_HIGHLIGHT_MNG_TIM', 'BUT_HIGHLIGHT_RETRO_COLOR',
    'BUT_LONG_SHARPIE_XANH', 'BUT_NUOC_CS_8623', 'BUT_XOA_NUOC',
    'HO_DOUBLE_8GM', 'KEP_BUOM_19MM', 'KEP_BUOM_25MM',
    'NGOI_CHI_MNG_0.5_100PCS', 'SO_TAY_A6', 'THUOC_CAMPUS_15CM',
    'THUOC_DO_DO', 'THUOC_PARABOL', 'XOA_KEO_CAPYBARA_9566'
]



class BYTETrackerArgs:
    def __init__(self, track_thresh, track_buffer, mot20, match_thresh,
                aspect_ratio_thresh, min_box_area):
        self.track_thresh        = track_thresh
        self.track_buffer        = track_buffer
        self.mot20               = mot20
        self.match_thresh        = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area        = min_box_area

def count_object(video_path):
    assert type(video_path)       == int, "video_path argument should be string"
    product_prices=    database.get_product_prices()

    args = BYTETrackerArgs(track_thresh= 0.4,
                          track_buffer= 30,
                          mot20=True,
                          match_thresh=0.8,
                          aspect_ratio_thresh=0.8,
                          min_box_area= 10)

    #Create attribute of transaction storing place:
    transaction_id = 0

    os.makedirs("../receipts", exist_ok=True)
    print(f"Starting transaction ID: {transaction_id}")

    obj_tracker = BYTETracker(args)
    cap = cv2.VideoCapture(video_path)

    # fps = vid.get(cv2.CAP_PROP_FPS)
    #dictionary for products

    class_counts = {name: 0 for name in product_prices.keys()}
    tracked_ids =  {}
    tracked_product_ids = set()  # Set to store IDs of counted products
    #
    # # FPS
    # fps_list = []  # Lưu trữ FPS tại mỗi frame
    # time_list = []  # Lưu trữ thời gian thực hiện mỗi frame

    class_name = ""

    while True:
        if not cap.isOpened():
            break
        else:
            #initiate transaction
            transaction_id += 1
            transaction_file_path = os.path.join("../receipts", f"transaction_{transaction_id}.txt")

            ret, frame  = cap.read()


            results      = model.predict(frame)

            first_result = results[0]
            labels       = first_result.names



            # Dictionaries define
            # Prepare lists to store detected bounding box coordinates and scores
            detections = []
            track_ids = []
            confidences=[]
            # List of detected products
            detected_products = []  # List to store names of detected products
            if ret:
                frame               = cv2.resize(frame, (640,480))


                orig_frame          = frame.copy() # we take a copy of our original frame before loosing it.
                #Get height, width of frame
                frame_h, frame_w    = frame.shape[:2]
                frame_size          = np.array([frame_h, frame_w])

                for result in results:
                    for box,r in zip(result.boxes, result.boxes.data):
                        x,y,w,h     = box.xywh[0]
                        class_id    = int(box.cls[0])
                        class_name  = labels[class_id] if labels[class_id] else ""

                        #Get products name - price
                        if class_name in product_prices:
                            price = product_prices[class_name]
                            print(f"Product: {class_name}, Price: {price}")

                        x1,y1,x2,y2 = int(x - w / 2), int(y - h / 2),int(x + w / 2),int(y + h / 2)

                        score = round(float(r[-2].item()), 2)  # Round to 2 decimal places
                        track_id= int(r[-1].item())

                        detections.append([x1, y1, x2, y2, score]) # xyxy and score

                        track_ids.append(track_id)
                        confidences.append(score)

                        # print(f"Detection: [{x1}, {y1}, {x2}, {y2},{score}]")
                        # print(f"Track id : {track_ids}")
                        # print(f"Confidence : {confidences}")

                #BYTE TRACK GENERATIONS
                if len(detections) > 0:
                    track_ids = obj_tracker.update(np.array(detections), frame_size, frame_size)

                    # Get tracking coordinate from YOLO     detections
                    for track in track_ids:
                        x1b, y1b, width_b, height_b = track.tlwh
                        x2b, y2b = x1b + width_b, y1b + height_b
                        track_id = track.track_id

                        if track_id not in tracked_ids:
                            tracked_ids[class_name]=set()
                            tracked_ids[class_name].add(track_id)  # Processed Track

                            if track_id not in tracked_product_ids:  # Ensure we don't count the same product again
                                class_counts[class_name] += 1
                                tracked_product_ids.add(track_id)  # Mark this product as counted

                        # Draw bbox of Byte-tracking frame
                        cv2.rectangle(orig_frame, (int(x1b), int(y1b)), (int(x2b), int(y2b)), (0, 255, 0), 5)

                        # ad
                        detected_products.append(class_name)
                        #  List of products in frame:
                        y_offset = 50
                        for product, count in class_counts.items():
                            if count > 0:
                                cv2.putText(orig_frame, f"{product}: {count}", (50, y_offset),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                                            thickness=2)
                                y_offset += 30
                        total_price = 0
                        for product, count in class_counts.items():
                            if count > 0:
                                price = product_prices.get(product, 0)
                                total_price += count * price
                                print(f"total price :{total_price}")

                        # Check to remove item
                        for track_id in list(tracked_ids):  # Create a clone list
                            if track_id not in [track.track_id for track in track_ids]:  # If track is not exist in new frame
                                # Reset all the counter till detected objects no longer exist in new frame
                                if track_id in class_counts:
                                    class_counts[class_name] = 0
                                tracked_ids.remove(track_id)

                cv2.imshow("Frame",orig_frame)
                # press esc for quitting the video

                total_price = 0

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    #Get date and time
                    current_time = datetime.datetime.now()
                    # Bill generator code:
                    with open(transaction_file_path,"w",encoding="utf-8") as file:
                        file.write(f"STATIONARY SHOP\nBill exported at {current_time}\n")
                        file.write(f"Transaction ID: {transaction_id}\n")
                        file.write("--------------------------------------------------\n")
                        for product, count in class_counts.items():
                            if count > 0:
                                price = product_prices.get(product, 0)
                                total_price += count * price

                                file.write(f"{product}  :\t\t{count} \t{price}\n\n")
                                file.write(f"total price:\t\t\t\t\t\t{total_price}\n")
                                file.write(f"Bill saved :\t\t\t\t{transaction_file_path}\n")
                                print("\n\n\n")
                                break
                    cap.release()
                    cv2.destroyAllWindows()
                    new_session = input("Do you want to start a new session? (y/n): ").strip().lower()
                    if new_session !="y":
                        print("Exiting program")
                        break
                    else:
                        print("Staring new seasons")
def main():
    count_object(0)

if __name__ == "__main__":
    main()

# if len(detections) > 0:
#     track_ids = obj_tracker.update(np.array(detections),frame_size,frame_size)
#     for track in track_ids:
#         # Extract bounding box in (top-left x, top-left y, width, height) format
#         x1b, y1b, width_b, height_b = track.tlwh
#         x2b, y2b = x1b + width_b, y1b + height_b  # Bottom-right corne
#         track_id = track.track_id
#
#         if track_id not in tracked_ids:
#             tracked_ids.add(track_id)  # Mark this track_id as processed
#             if track_id is not None:
#                 class_counts[class_name] += 1  # Increment the count for the detected class
#         # Draw bounding box on the Byte-tracking-frame
#         cv2.rectangle(orig_frame, (int(x1b), int(y1b)), (int(x2b), int(y2b)), (0, 255, 0),
#                       2)
#
#         detected_products.append(class_name)  # Add detected product name to list
