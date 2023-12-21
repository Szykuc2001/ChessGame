'''
Install instructions:
1. Put the python file and mp4 file in the same directory
2. Install the required dependencies
3. Use a mobile app that allows using a phone as camera, if you lack one in computer
4. Start the program

s22466 Szymon Kuczyński
'''
from mtcnn import MTCNN
import cv2
import time

'''
Utilizes the MTCNN detector to detect faces within the frame.
Confidence Check: Verifies if the confidence level of the detected face is higher than 0.9 (90%). Confidence denotes the reliability of the face detection.
Eye Keypoints Extraction: Retrieves the keypoints for the left and right eyes if the confidence level surpasses the threshold.
Return:
If both left and right eye keypoints are detected with high confidence, the function returns a boolean True to indicate eyes detection along with the coordinates of the left and right eyes.
If the confidence threshold is not met or no eyes are detected, the function returns a boolean False and None values for eye coordinates.
'''
def detect_eyes(frame):
    detector = MTCNN()
    faces = detector.detect_faces(frame)

    for face in faces:

        if face['confidence'] > 0.9:
            left_eye = face['keypoints']['left_eye']
            right_eye = face['keypoints']['right_eye']

            return True, left_eye, right_eye

    return False, None, None

'''
Camera Initialization:

cv2.VideoCapture(0): Initializes the camera feed using OpenCV. The argument 0 signifies the default camera.
video_file = 'video.mp4': Specifies the video file name.
video = cv2.VideoCapture(video_file): Initializes another video capture object for the specified video file.
Window Naming:

cv2.namedWindow('Camera') and cv2.namedWindow('Video'): Creates two separate windows named 'Camera' and 'Video' using OpenCV for displaying camera and video feed, respectively.
Variables Initialization:

last_seen_time = time.time(): Initializes a variable to store the last time the user was seen looking at the screen.
alert_displayed = False: Tracks whether the alert message is currently displayed.
is_playing = False: Indicates if the video is currently playing or paused.
paused_frame = 0: Keeps track of the frame number where the video was paused.
pause_start_time = 0: Records the time when the pause started.
Alert Message:

alert_message = "Alert: User not looking at the screen": Defines the alert message that will be displayed when the system detects the user not looking at the screen.

Reading Camera Frame:

ret, camera_frame = camera.read(): Reads a frame from the camera feed and stores it in camera_frame. The variable ret indicates whether the frame was successfully read.
Eyes Detection:

eyes_detected, left_eye, right_eye = detect_eyes(camera_frame): Calls the detect_eyes function to identify eyes in the current camera frame. This function returns a boolean eyes_detected along with the coordinates of the left and right eyes.
Actions based on Eye Detection:

if eyes_detected:: If eyes are detected in the frame:

if not is_playing:: Checks if the video is not playing:
video.set(cv2.CAP_PROP_POS_FRAMES, paused_frame): Sets the video frame to the position where it was paused.
is_playing = True: Updates the is_playing flag to indicate that the video is now playing.
last_seen_time = time.time(): Records the time when eyes were last detected, indicating the user is looking at the screen.
if alert_displayed:: Resets alert_displayed to False if an alert was previously displayed.
pause_start_time = 0: Resets the pause timer if eyes are detected.
else: (if eyes are not detected):

if is_playing:: Checks if the video is currently playing:
is_playing = False: Updates is_playing to False, pausing the video.
paused_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES)): Stores the current frame number as the paused frame.
alert_displayed = True: Sets alert_displayed to True to display the alert.
if pause_start_time == 0:: Starts the pause timer if it was not already initiated.
pause_start_time = time.time(): Records the time when the pause started.
Displaying Camera Frame:

cv2.imshow('Camera', camera_frame): Displays the camera frame in the 'Camera' window created earlier.

Video Playback:

if is_playing:: Checks if the video is currently playing.
ret, video_frame = video.read(): Reads a frame from the video.
if ret:: If a frame is successfully read:
cv2.imshow('Video', video_frame): Displays the video frame in the 'Video' window.
else:: If no frame is read (video ends):
break: Exits the loop as the video has ended.
Alert Display:

if alert_displayed and not is_playing:: Checks if an alert needs to be displayed while the video is paused.
if time.time() - pause_start_time > 5:: Checks if the video has been paused for more than 5 seconds.
text_size = cv2.getTextSize(...): Calculates the size of the alert text for positioning.
cv2.putText(video_frame, alert_message, ...): Adds the alert message to the center of the video frame.
cv2.imshow('Video', video_frame): Displays the video frame with the alert message.
Exit Condition:

if cv2.waitKey(1) & 0xFF == ord('q'):: Checks if the 'q' key is pressed.
break: If 'q' is pressed, exits the loop, terminating the program.
'''
def main():
    camera = cv2.VideoCapture(0)
    video_file = 'video.mp4'
    video = cv2.VideoCapture(video_file)

    cv2.namedWindow('Camera')
    cv2.namedWindow('Video')

    last_seen_time = time.time()
    alert_displayed = False
    is_playing = False
    paused_frame = 0
    pause_start_time = 0

    alert_message = "Alert: User not looking at the screen"


    while True:
        ret, camera_frame = camera.read()

        eyes_detected, left_eye, right_eye = detect_eyes(camera_frame)

        if eyes_detected:
            if not is_playing:
                video.set(cv2.CAP_PROP_POS_FRAMES, paused_frame)
                is_playing = True

            last_seen_time = time.time()
            if alert_displayed:
                alert_displayed = False

            pause_start_time = 0

        else:
            if is_playing:

                is_playing = False
                paused_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                alert_displayed = True

                if pause_start_time == 0:
                    pause_start_time = time.time()

        cv2.imshow('Camera', camera_frame)

        if is_playing:
            ret, video_frame = video.read()
            if ret:
                cv2.imshow('Video', video_frame)
            else:
                break

        if alert_displayed and not is_playing:

            if time.time() - pause_start_time > 5:

                text_size = cv2.getTextSize(alert_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                video_frame_height, video_frame_width, _ = video_frame.shape
                text_x = (video_frame_width - text_size[0]) // 2
                text_y = (video_frame_height + text_size[1]) // 2
                cv2.putText(video_frame, alert_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow('Video', video_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
