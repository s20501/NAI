# Authors: Marcin Żmuda-Trzebiatowski and Jakub Cirocki
# Example: https://github.com/s20501/NAI/blob/main/zjazd_4/example.PNG
#
# A program that recognizes gestures and overlaps certain images on the user's face depending on that gesture.

import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense
import os



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic # Holistic model
face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                         min_detection_confidence=0.5,min_tracking_confidence=0.3)


sequence = []
sentence = []
predictions = []
threshold = 0.9

model = Sequential()

actions = np.array(['first', 'second', 'third', 'none'])


predition = 'none'


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def train_model():
    DATA_PATH = os.path.join('MP_Data')
    sequence_length = 30
    label_map = {label: num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.load_weights('action.h5')
    # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # model.fit(X_train, y_train, epochs=500)
    return model

def detectFacialLandmarks(image, face_mesh):
    '''
    This function performs facial landmarks detection on an image.
    Args:
        image:     The input image of person(s) whose facial landmarks needs to be detected.
        face_mesh: The face landmarks detection function required to perform the landmarks detection.
        display:   A boolean value that is if set to true the function displays the original input image,
                   and the output image with the face landmarks drawn and returns nothing.
    Returns:
        output_image: A copy of input image with face landmarks drawn.
        results:      The output of the facial landmarks detection on the input image.
    '''

    if face_mesh is not None:
        # Perform the facial landmarks detection on the image, after converting it into RGB format.
        results = face_mesh.process(image[:, :, ::-1])

        # Create a copy of the input image to draw facial landmarks.
        output_image = image[:, :, ::-1].copy()

        # face = np.array([[res.x, res.y, res.z] for res in
        #                  results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

        # Check if facial landmarks in the image are found.
        if results.multi_face_landmarks:

            # Iterate over the found faces.
            for face_landmarks in results.multi_face_landmarks:
                # Draw the facial landmarks on the output image with the face mesh tesselation
                # connections using default face mesh tesselation style.
                mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Draw the facial landmarks on the output image with the face mesh contours
                # connections using default face mesh contours style.
                mp_drawing.draw_landmarks(image=output_image, landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())


        return np.ascontiguousarray(output_image[:, :, ::-1], dtype=np.uint8), results
    return None


def getSize(image, face_landmarks, INDEXES):
    '''
    This function calculate the height and width of a face part utilizing its landmarks.
    Args:
        image:          The image of person(s) whose face part size is to be calculated.
        face_landmarks: The detected face landmarks of the person whose face part size is to
                        be calculated.
        INDEXES:        The indexes of the face part landmarks, whose size is to be calculated.
    Returns:
        width:     The calculated width of the face part of the face whose landmarks were passed.
        height:    The calculated height of the face part of the face whose landmarks were passed.
        landmarks: An array of landmarks of the face part whose size is calculated.
    '''

    # Retrieve the height and width of the image.
    image_height, image_width, _ = image.shape

    # Convert the indexes of the landmarks of the face part into a list.
    INDEXES_LIST = list(itertools.chain(*INDEXES))

    # Initialize a list to store the landmarks of the face part.
    landmarks = []

    # Iterate over the indexes of the landmarks of the face part.
    for INDEX in INDEXES_LIST:
        # Append the landmark into the list.
        landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                          int(face_landmarks.landmark[INDEX].y * image_height)])

    # Calculate the width and height of the face part.
    _, _, width, height = cv2.boundingRect(np.array(landmarks))

    # Convert the list of landmarks of the face part into a numpy array.
    landmarks = np.array(landmarks)

    # Retrurn the calculated width height and the landmarks of the face part.
    return width, height, landmarks

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=True):
    '''
    This function will overlay a filter image over a face part of a person in the image/frame.
    Args:
        image:          The image of a person on which the filter image will be overlayed.
        filter_img:     The filter image that is needed to be overlayed on the image of the person.
        face_landmarks: The facial landmarks of the person in the image.
        face_part:      The name of the face part on which the filter image will be overlayed.
        INDEXES:        The indexes of landmarks of the face part.
        display:        A boolean value that is if set to true the function displays
                        the annotated image and returns nothing.
    Returns:
        annotated_image: The image with the overlayed filter on the top of the specified face part.
    '''

    # Create a copy of the image to overlay filter image on.
    annotated_image = image.copy()

    # Errors can come when it resizes the filter image to a too small or a too large size .
    # So use a try block to avoid application crashing.
    try:

        # Get the width and height of filter image.
        filter_img_height, filter_img_width, _ = filter_img.shape

        # Get the height of the face part on which we will overlay the filter image.
        _, face_part_height, landmarks = getSize(image, face_landmarks, INDEXES)

        # Specify the height to which the filter image is required to be resized.
        required_height = int(face_part_height * 1.5)

        # Resize the filter image to the required height, while keeping the aspect ratio constant.
        resized_filter_img = cv2.resize(filter_img, (int(filter_img_width *
                                                         (required_height / filter_img_height)),
                                                     required_height))

        # Get the new width and height of filter image.
        filter_img_height, filter_img_width, _ = resized_filter_img.shape

        # Convert the image to grayscale and apply the threshold to get the mask image.
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)

        # Calculate the center of the face part.
        center = landmarks.mean(axis=0).astype("int")

        # Check if the face part is mouth.
        if face_part == 'MOUTH':

            # Calculate the location where the smoke filter will be placed.
            location = (int(center[0] - filter_img_width / 3), int(center[1]))

        # Otherwise if the face part is an eye.
        else:

            # Calculate the location where the eye filter image will be placed.
            location = (int(center[0] - filter_img_width / 2), int(center[1] - filter_img_height / 2))

        # Retrieve the region of interest from the image where the filter image will be placed.
        ROI = image[location[1]: location[1] + filter_img_height,
              location[0]: location[0] + filter_img_width]

        # Perform Bitwise-AND operation. This will set the pixel values of the region where,
        # filter image will be placed to zero.
        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)

        # Add the resultant image and the resized filter image.
        # This will update the pixel values of the resultant image at the indexes where
        # pixel values are zero, to the pixel values of the filter image.
        resultant_image = cv2.add(resultant_image, resized_filter_img)

        # Update the image's region of interest with resultant image.
        annotated_image[location[1]: location[1] + filter_img_height,
        location[0]: location[0] + filter_img_width] = resultant_image

    # Catch and handle the error(s).
    except Exception as e:
        pass

    # Check if the annotated image is specified to be displayed.
    if display:

        # Display the annotated image.
        plt.figure(figsize=[10, 10])
        plt.imshow(annotated_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the annotated image.
        return annotated_image


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def switch(image, prediction, face_landmarks):
    if prediction == 'first':
        return overlay(image, cv2.imread('assets/ping.png'), face_landmarks,'FACE',
                         mp_holistic.FACEMESH_TESSELATION, display=False)
    elif prediction == 'second':
        return overlay(image, cv2.imread('assets/ping2.png'), face_landmarks,'FACE',
                        mp_holistic.FACEMESH_TESSELATION, display=False)
    elif prediction == 'third':
        return overlay(image, cv2.imread('assets/unknown.png'), face_landmarks,'FACE',
                        mp_holistic.FACEMESH_TESSELATION, display=False)
    else:
        return image


model = train_model()
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
res = None
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    # frame.flags.writeable = False
    # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, results = mediapipe_detection(frame, holistic)

    _, face_mesh_results = detectFacialLandmarks(frame, face_mesh_videos)

    # draw_styled_landmarks(image, results)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        # print(actions[np.argmax(res)])
        predictions.append(np.argmax(res))

        # 3. Viz logic
        if np.unique(predictions[-10:])[0] == np.argmax(res):
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5:
            sentence = sentence[-5:]


    # Draw the face mesh annotations on the image.
    # image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.face_landmarks and face_mesh_results is not None:
      for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
          if res is not None:
            image = switch(image, actions[np.argmax(res)], face_landmarks)
        # 2. Prediction logic
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


