import cv2
import numpy as np
import os
import uuid

class ObjectDetectionModel:
    def __init__(self, config_path, weights_path, names_path):
        """
        Initializes the YOLO model by loading its configuration, weights, and class names.
        """
        if not os.path.exists(config_path) or not os.path.exists(weights_path) or not os.path.exists(names_path):
            raise FileNotFoundError("YOLO model files not found. Please ensure they are in the 'yolo_model' directory.")

        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.output_layers = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        print("YOLO model successfully loaded.")

    def detect_defects(self, image):
        """
        Detects objects in an input image and returns a list of detected defects.
        
        Args:
            image (numpy.ndarray): The input image.
            
        Returns:
            list: A list of detected defects, each as a dictionary.
        """
        height, width, _ = image.shape
        
        # Prepare the image for the model (blob)
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run the model to get the predictions
        layer_outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Process the outputs to find bounding boxes and confidence scores
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5: # Filter out weak predictions
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression to remove redundant boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        defects = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                defect = {
                    "defectID": str(uuid.uuid4()),
                    "type": self.classes[class_ids[i]],
                    "boundingBox": {
                        "x": boxes[i][0],
                        "y": boxes[i][1],
                        "width": boxes[i][2],
                        "height": boxes[i][3]
                    },
                    "confidence": confidences[i]
                }
                defects.append(defect)
                
        return defects