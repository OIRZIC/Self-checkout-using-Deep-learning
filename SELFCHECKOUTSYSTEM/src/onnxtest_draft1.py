# Some standard imports
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

# import onnxruntime
# print(onnxruntime.__version__)

import cv2
import numpy as np
import onnxruntime as ort  # Thêm import này để sử dụng 'ort' thay cho 'onnxruntime'
import cv2
import numpy as np
from torch.ao.nn.quantized.functional import threshold
model_path= "../weights/best_v8m.onnx"

# labels = [
#     "BUT_CHI_DIXON",
#     "BUT_HIGHLIGHT_MNG_TIM",
#     "BUT_HIGHLIGHT_RETRO_COLOR",
#     "BUT_LONG_SHARPIE_XANH",
#     "BUT_NUOC_CS_8623",
#     "BUT_XOA_NUOC",
#     "HO_DOUBLE_8GM",
#     "KEP_BUOM_19MM",
#     "KEP_BUOM_25MM",
#     "NGOI_CHI_MNG_0.5_100PCS",
#     "SO_TAY_A6",
#     "THUOC_CAMPUS_15CM",
#     "THUOC_DO_DO",
#     "THUOC_PARABOL",
#     "XOA_KEO_CAPYBARA_9566"
# ]

# prefer CUDA Execution Provider over CPU Execution Provider
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# initialize the model.onnx
session = ort.InferenceSession(model_path, providers=EP_list)

# Danh sách tên lớp và các màu sắc
class_names = [
    "BUT_CHI_DIXON",
    "BUT_HIGHLIGHT_MNG_TIM",
    "BUT_HIGHLIGHT_RETRO_COLOR",
    "BUT_LONG_SHARPIE_XANH",
    "BUT_NUOC_CS_8623",
    "BUT_XOA_NUOC",
    "HO_DOUBLE_8GM",
    "KEP_BUOM_19MM",
    "KEP_BUOM_25MM",
    "NGOI_CHI_MNG_0.5_100PCS",
    "SO_TAY_A6",
    "THUOC_CAMPUS_15CM",
    "THUOC_DO_DO",
    "THUOC_PARABOL",
    "XOA_KEO_CAPYBARA_9566"
]
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

# Các hàm liên quan đến phát hiện và vẽ kết quả
def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []

    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)
    keep_boxes = []

    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]
        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    return intersection_area / union_area


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]
        draw_box(det_img, box, color)
        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box(image, box, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image, text, box, color=(0, 0, 255), font_size=0.001, text_thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)
    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)
    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)


def draw_masks(image, boxes, classes, mask_alpha=0.3):
    mask_img = image.copy()
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


# Lớp YOLOv8
class YOLOv8:

    def __init__(self, path, conf_thres=0.8, iou_thres=0.6):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = ort.InferenceSession(path, providers=ort.get_available_providers())
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


# Main Function to Run Object Detection on Video
if __name__ == '__main__':
    model_path = r"C:\Users\nguye\Desktop\iot\CV_GPU\fall_detection\trained_model\a1\detect\train\weights\best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.8, iou_thres=0.6)

    cap = cv2.VideoCapture(
        r'C:\Users\nguye\Desktop\iot\CV_GPU\fall_detection\exp9\New folder\Fall_Detection_Using_Yolov8-main\50way.mp4'
    )
    # cap = cv2.VideoCapture(0)
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        if cv2.waitKey(1) == ord('q'):
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        boxes, scores, class_ids = yolov8_detector(frame)

        # # Set breath_beat based on detection of 'fallen-person'
        # breath_beat = any(class_names[class_id] == 'fallen-person' for class_id in class_ids)

        # Draw the detection results on the frame
        combined_img = yolov8_detector.draw_detections(frame)

        # Display the result
        cv2.imshow("Detected Objects", combined_img)

        # Optional: Print the breath_beat status for debugging
        # print("breath_beat:", breath_beat)

    cap.release()
    cv2.destroyAllWindows()

