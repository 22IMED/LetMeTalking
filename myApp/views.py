from django.shortcuts import render

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp        
from keras.models import Sequential ,load_model
from keras.layers import LSTM, Dense, SpatialDropout1D
from keras.callbacks import TensorBoard
from django.http import HttpResponseBadRequest


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def most_frequent(List):
    return max(set(List), key = List.count)
@csrf_exempt
def process_video(request):
    if request.method == 'POST' and request.FILES['video']:
        #print(request.FILES['video'])
        
        
        mp_holistic = mp.solutions.holistic # Holistic model
        mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        #model=model_()
        model = load_model('MyModel/action_model_ .h5')
        #model.load_weights('C:/Users/DELL/Desktop/action_model.h5')
        
        sentence =[]
        sequence=[]
        sentence2=[]
        threshold = 0.98
        #actions1=np.array(['hello','please','thank you',"me & I'm",'learn','no','yes','finish','nice to meet you',"how are you"])
        #actions1=np.array(['hello','help','thank you',"no",'learn','yes','please','finish',"how are you",'nice to meet you'])
        actions1=['Bye','Good','GoTo','Hi','HowAreYou','NiceToMeetYou','No','Please','ThankYou','Yes']
        label_map = {label:num for num, label in enumerate(actions1)}

        #cap = cv2.VideoCapture(video_file.temporary_file_path())
        # Set mediapipe model 
        i=0
        video_file = request.FILES['video']
        #cap = cv2.VideoCapture(video_file.temporary_file_path())

        video = request.FILES['video']
        with open('video.mp4', 'wb+') as f:
            for chunk in video.chunks():
                f.write(chunk)
        # extract frames from the video using OpenCV
        cap = cv2.VideoCapture('video.mp4')

        ret, frame = cap.read()
        res=[]
        window = []
        sequences=[]
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while ret:
                i+=1
                #print(i)
                # Read feed
                if ret:
                  

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    #print(results)
                    
                    # Draw landmarks
                    #draw_styled_landmarks(image, results)
                    
                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)
            #         sequence.insert(0,keypoints)
            #         sequence = sequence[:30]
                    sequence.append(keypoints)
                    window.append(keypoints)
                    #sequence = sequence[-59:]
                    print(len(sequence))
                    
                    if len(sequence) == 59:
                        print('pridection ....')
                        sequences.append(window)
                        X = np.array(sequences)
                        res = model.predict(X)
                        #sentence8=[]
                        #sentence8.append(actions1[np.argmax(res)])
                        #res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        #print(actions1[np.argmax(res)])
                        #print(np.max(res))
                        
                        
                    #3. Viz logic
                    ''' if res[np.argmax(res)] >threshold: 
                            sentence2.append(actions1[np.argmax(res)])
                            if len(sentence) > 0: 
                                if actions1[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions1[np.argmax(res)])
                            else:
                                sentence.append(actions1[np.argmax(res)])'''

                        #if len(sentence) > 30: 
                           # sentence = sentence[-30:]

                
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                ret, frame = cap.read()    
            cap.release()
            cv2.destroyAllWindows()
        
        # process the video file here
        return JsonResponse({'message':[actions1[np.argmax(res)]]})
    else:
        return JsonResponse({'error': 'Invalid request'})