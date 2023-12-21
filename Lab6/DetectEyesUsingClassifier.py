'''
Install instructions:
1. Put the python file and mp4 file in the same directory
2. Install the required dependencies
3. Use a mobile app that allows using a phone as camera, if you lack one in computer
4. Start the program

s22466 Szymon Kuczyński
'''
import cv2
import time

'''
Convert Frame to Grayscale:

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY): Converts the input frame (which is typically in BGR color format) to grayscale. Grayscale simplifies image processing as it represents the image in terms of intensity rather than colors.
Eye Cascade Classifier Initialization:

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml'): Initializes the Haar Cascade Classifier specifically designed for detecting eyes. The cv2.CascadeClassifier is provided with the path to the pre-trained XML file (haarcascade_eye.xml) containing the necessary information to detect eyes.
Eye Detection:

eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5): Utilizes the detectMultiScale function of the cascade classifier to identify potential eye regions in the grayscale frame.
gray: Grayscale frame used as input for eye detection.
scaleFactor=1.1: Parameter specifying how much the image size is reduced at each image scale. It compensates for the eye size's variation concerning the distance from the camera.
minNeighbors=5: Parameter specifying how many neighbors each candidate rectangle should have to retain it. Higher values reduce false positives but might miss some eyes.
Return Statement:

return len(eyes) > 0: Returns True if at least one eye is detected in the frame, otherwise False. It evaluates the number of detected eye regions. If there's at least one region identified as an eye, it returns True; otherwise, it returns False.
'''
def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(eyes) > 0

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

        if detect_eyes(camera_frame):
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
