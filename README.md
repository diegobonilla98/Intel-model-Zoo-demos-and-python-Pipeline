# Intel-model-Zoo-demos-and-python-Pipeline
My first build of the Intel models with OpenVINO and the pipeline for python.

Download OpenVINO toolkit in https://software.intel.com/en-us/openvino-toolkit. Then an important step it to make all the environment variables "forever" by manually adding them.
Then the python pipeline just executes the demo using the subprocess library and reads from it and (poorly and messy) parses the info and separates it into people every one with a dictionary with all the data: age, emotion, position and gender. (NEEDS REUPLOAD FOR MULTIPLE FACES ARRAY ERROR)
A huge problem is that uses the webcam so no other program can use it. I'm trying to recreate all the neural models with free databases so I can manipulate them better.

This images are a couple of awkward screenshots of the head_pose_estimator+age+emo+gender and the gaze estimator:

![Head demo] (intel_face_demo.png)
![Gaze demo] (intel_gaze_demo.png)
