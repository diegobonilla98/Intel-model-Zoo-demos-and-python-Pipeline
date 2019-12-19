import subprocess
import re
import shlex
import numpy as np

video_stream_format = 'cam'
procExe = subprocess.Popen(shlex.split('interactive_face_detection_demo.exe -i ' + video_stream_format +
                                       ' -m "C:/Program Files ('
                                       'x86)/IntelSWTools/openvino/deployment_tools/tools/model_downloader/intel/face'
                                       '-detection-adas-0001/FP16/face-detection-adas-0001.xml" -m_ag "C:/Program '
                                       'Files ('
                                       'x86)/IntelSWTools/openvino/deployment_tools/tools/model_downloader/intel/age'
                                       '-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml" '
                                       '-m_em "C:/Program Files ('
                                       'x86)/IntelSWTools/openvino/deployment_tools/tools/model_downloader/intel'
                                       '/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml" '
                                       '-m_hp "C:/Program Files ('
                                       'x86)/IntelSWTools/openvino/deployment_tools/tools/model_downloader/intel/head'
                                       '-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml" -r '
                                       '-no_show -no_wait'), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)

search = False
print_info = False
emotions_decode = ('neutral', 'happy', 'sad', 'surprise', 'anger')

session_faces = []
sessions = []

age_info = yaw_info = pitch_info = roll_info = 0.0
gender_info = emotion_info = ""

face_idx = 0
while procExe.poll() is None:
    line = procExe.stdout.readline()[:-1]

    male_prob = re.search(r"prob = (\d*\.*\d*)", line)
    pos = re.search(r"(\(-*\d*,-*\d*\))-(\(-*\d*,-*\d*\))", line)
    age = re.search(r"age = (\d*\.*\d*)", line)
    emotions = re.search(r"neutral = (\d*\.*\d*)", line)
    head_pose = re.search(r"yaw = (\d*\.*\d*)", line)

    if search:
        if emotions:
            emotions_prob = [em.split(' ')[2] for em in emotions.string.split(', ')]
            emotion_info = emotions_decode[int(np.argmax(emotions_prob))]
            if print_info:
                print(emotion_info)
        if head_pose:
            yaw_info = float(line[head_pose.start():].split(', ')[0].split(' ')[2])
            pitch_info = float(line[head_pose.start():].split(', ')[1].split(' ')[2])
            roll_info = float(line[head_pose.start():].split(', ')[2].split(' ')[2])
            if print_info:
                print("Yaw:", yaw_info, "\tPitch:", pitch_info, "\tRoll:", roll_info)

            face_dict = {'gender:': gender_info, 'age:': age_info, 'yaw:': yaw_info, 'pitch': pitch_info,
                         'roll': roll_info, 'emotion': emotion_info, 'position': None}
            session_faces.append(face_dict)
            search = False
            continue

    if not pos and male_prob:
        search = True
        prob = float(male_prob.group().split(' ')[2])
        gender_info = "male" if prob > 0.5 else "female"
        if print_info:
            print(gender_info)
        if age:
            age_info = float(age.group().split(' ')[2])
            if print_info:
                print("Age:", age_info)
    if line.find('WILL BE RENDERED!') != -1:
        # print(len(session_faces))
        if len(session_faces) > 0:
            head_pos_init = pos.group().find(')')
            head_pos_x = eval(pos.group()[:head_pos_init+1])
            head_pos_y = eval(pos.group()[head_pos_init+2:])

            session_faces[face_idx]['position'] = (head_pos_x, head_pos_y)
            face_idx += 1
            # print(session_faces[face_idx])
    elif len(session_faces) > 0:
        print(session_faces, '\n')
        face_idx = 0
        session_faces = []
