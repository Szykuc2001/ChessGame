Two apps that play a video if user is looking on the screen and stop if they are not looking at the screen.
First solution uses classifier to determine if eyes are present in the frame from camera, it is less accurate but video plays in higher framerate.
Second solution uses MTCNN to determine wether the eyes are in the frame from camera or not, it is much more accurate but framerate is much lower than in classifier.
