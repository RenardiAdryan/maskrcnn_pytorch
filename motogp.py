
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import random
import warnings


model = torch.load('model\mask-rcnn-motogp-10-20220813-004407.pt',map_location="cpu")
# set to evaluation mode
model.eval()
CLASS_NAMES = ['__background__', 'motogp']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0
    
    """
    
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    img = img.to(device)
    pred = model([img])

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
    if pred_t:
      pred_t = pred_t[-1]
      masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
      if masks.ndim<3:
        masks = np.expand_dims(masks, axis=0)
      pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
      pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
      masks = masks[:pred_t+1]
      pred_boxes = pred_boxes[:pred_t+1]
      pred_class = pred_class[:pred_t+1]
      return masks, pred_boxes, pred_class
    else:
      masks=[]
      pred_boxes=[] 
      pred_class=[]
      return masks, pred_boxes, pred_class




rect_th=2
text_size=0.5
text_th=2

video = cv2.VideoCapture("Motogp\motogp2020.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

name = "MOTOGP_Segementation" + '.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(name, fourcc, 25.0, (width, height))
count_error=0
while True:
    ret, frame = video.read()
    try:
        confidence = 0.9
        masks, boxes, pred_cls = get_prediction(frame, confidence)
        masks = np.array(masks)
        img = frame.copy()
        if len(masks)!=0 and len(boxes)!=0 and len(pred_cls)!=0:
            boxes, masks, pred_cls=zip(*sorted(zip(boxes, masks, pred_cls),key=lambda x: x[0][0][1],reverse=True))
            for i in range(len(masks)):
                rgb_mask = get_coloured_mask(masks[i])
                img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                cv2.rectangle(img, (int(boxes[i][0][0]),int(boxes[i][0][1])), (int(boxes[i][1][0]),int(boxes[i][1][1])),color=(0, 255, 0), thickness=rect_th)
                cv2.putText(img,pred_cls[i]+" "+str(i+1), (int(boxes[i][0][0]),int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

        cv2.imshow("MOTOGP_Segementation",img)
        out.write(img)
    except Exception as e:
      count_error = count_error+1
      print(e)
      cv2.imwrite("MOTOGP_Segementation_error_"+str(count_error)+"_.jpg", frame)

    key = cv2.waitKey(25)
    if key == 27:
        break

out.release()
video.release()
cv2.destroyAllWindows()
