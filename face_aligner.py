from skimage import transform
import cv2
import numpy as np
from imageio import imwrite, imsave
from operator import add
DESIRED_X = 64
DESIRED_Y = 42
DESIRED_SIZE = 48

FINAL_IMAGE_WIDTH = 128
FINAL_IMAGE_HEIGHT = 128

face_cascade = cv2.CascadeClassifier(
    'haar/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(
    'haar/Mouth.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

def get_pos(x, y, h, w):
    """
    Calculate the center coordinate of a rectangle 
    from a point, width & height
    """
    return np.array([x + (w/2), y + (h/2)])


def debug_landmarks(img, r_eye, l_eye, mouth):
    '''
    For debugging purposes, displaying the relevant landmarks in real time.
    '''
    r_eye = [int(x) for x in list(r_eye)]
    l_eye = [int(x) for x in list(l_eye)]
    mouth = [int(x) for x in list(mouth)]

    cv2.circle(img, tuple(l_eye), 10, (0, 0, 255), -1)
    cv2.circle(img, tuple(r_eye), 10, (0, 255, 0), -1)
    cv2.circle(img, tuple(mouth), 10, (255, 0, 0), -1)

    cv2.namedWindow("Landmarks")
    cv2.imshow("Landmarks", img)


def get_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_np = np.array(img)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return
    x,y,w,h = faces[0]
    mouth = mouth_cascade.detectMultiScale(gray, 1.2, 5)

    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    if eyes[0][0] < eyes[1][0]:
        eyes = eyes[::-1]
    if len(mouth) == 0 or len(eyes) < 2:
        return
    mouth_pos = get_pos(*mouth[0])
    right_eye_pos = get_pos(*eyes[0]) + np.array([x,y])
    left_eye_pos  = get_pos(*eyes[1]) + np.array([x,y])
    
    debug_landmarks(img,right_eye_pos,left_eye_pos,mouth_pos)
    return align(image_np, right_eye_pos, left_eye_pos, mouth_pos)


def align(image_np, right_eye_pos, left_eye_pos, mouth_pos):
    '''
    Rotate and align the image so the face is centered and aligned horizontally
    '''
    central_position = (left_eye_pos+right_eye_pos)/2
    face_width = np.linalg.norm(left_eye_pos - right_eye_pos)
    face_height = np.linalg.norm(central_position - mouth_pos)

    # Validate face width/height ratio
    if face_height * 0.7 <= face_width <= face_height * 1.5:
        face_size = (face_width + face_height) / 2
        # Rotate image so the eye-level is horizontal
        to_scale_factor = face_size / DESIRED_SIZE
        to_X_shift, to_Y_shift = central_position
        to_rotate_factor = np.arctan2(
            right_eye_pos[1] - left_eye_pos[1],
            right_eye_pos[0] - left_eye_pos[0])

        rotateT = transform.SimilarityTransform(
            scale=to_scale_factor, rotation=to_rotate_factor,
            translation=(to_X_shift, to_Y_shift))

        moveT = transform.SimilarityTransform(
            scale=1, rotation=0, translation=(-DESIRED_X, -DESIRED_Y))

        output_arr = transform.warp(image=image_np, inverse_map=(
            moveT + rotateT))[0:FINAL_IMAGE_HEIGHT, 0:FINAL_IMAGE_WIDTH]

        return output_arr