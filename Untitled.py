#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import threading
import ipywidgets as widgets
import cv2
import numpy as np
import pickle
import imutils
from IPython.display import display,Image


# In[2]:


addDatasetBtn = widgets.Button(
    description='Add Dataset',
    disabled=False,
    button_style='',
    tooltip='',
    icon='plus'
)
embedBtn = widgets.Button(
    description='Embedding',
    disabled=False,
    button_style='',
    tooltip='Extract Embedding.',
    icon='flask'
)
trainBtn = widgets.Button(
    description='Train',
    disabled=False,
    button_style='',
    tooltip='Train Model.',
    icon='train'
)
stopBtn = widgets.ToggleButton(
    value=False,
    description='Stop',
    disabled=False,
    button_style='danger', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Description',
    icon='square' # (FontAwesome names without the `fa-` prefix)
)
detectSlider = widgets.IntSlider(
    value=60,
    min=1,
    max=100,
    step=1,
    description='Threshold:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
faceSlider = widgets.IntSlider(
    value=60,
    min=1,
    max=100,
    step=1,
    description='Face:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

btnBox = widgets.HBox([embedBtn,trainBtn,stopBtn])
sliderBox = widgets.VBox([detectSlider,faceSlider])


# In[6]:


def refreshData():
    datalist = os.listdir('open-face-recognition/dataset')
    return datalist
def loadData():
    protoPath = os.path.sep.join("face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join("face_detection_model",
        "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open("output/le.pickle", "rb").read())
    
def cameraView():
    cap = cv2.VideoCapture(0)
    display_handle=display(None, display_id=True)
    i = 0
    while True:
        _, frame = cap.read()
        # frame = cv2.flip(frame, 1) # if your camera reverses your image
        
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        _, frame = cv2.imencode('.jpeg', frame)
        display_handle.update(Image(data=frame.tobytes()))
        if stopBtn.value==True:
            cap.release()
            display_handle.update(None)
            
def runEmbedding(b):
    embedBtn.disabled=True
    get_ipython().system('python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7')
    embedBtn.disabled=False
    pass
embedBtn.on_click(runEmbedding)
def runTraining(b):
    trainBtn.disabled=True
    get_ipython().system('python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle')
    trainBtn.disabled=False
    pass
trainBtn.on_click(runTraining)


# In[7]:


# Run
loadData()
thread = threading.Thread(target=cameraView, args=())
thread.start()
display(btnBox,sliderBox)


# In[ ]:





# In[ ]:




