from mmaction.apis import inference_recognizer, init_recognizer
import time
import cv2
import numpy as np
import os
import shutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

detection_list = []
texting_values = []
looking_values = []
talking_values = []

talk_high_prefix = "database/with_phone/high_pos/talk/trim/"
talk_high_num = 8
no_high_prefix = "database/no_phone/high_pos/"  # 17 videos
no_high_num = 17
low_trim_talk = "database/with_phone/low_pos/talk/"  # 7 videos
num_low_trim_talk = 7
no_low_prefix = "database/no_phone/low_pos/"  # 30 videos
no_low_num = 30

text_low_prefix = "database/with_phone/low_pos/text/"
text_low_num = 8
text_high_prefix = "database/with_phone/high_pos/text/"
text_high_num = 12

phone_high_prefix = "database/with_phone/high_pos/"  # 22 videos
phone_high_num = 22
text_high_prefix = "database/with_phone/high_pos/text/"  # 12 videos
text_high_num = 12

false_detect = "database/no_phone/high_pos/talk_false_detection/8_Trim_"

# global avg_text
global avg_look
global avg_talk



def get_videos():
    videos = []

    # Talk High
    # for i in range(1, talk_high_num+1, 1):
    #     video_name = talk_high_prefix + str(i) + "_Trim.mp4"
    #     videos.append(video_name)

    # # No High
    # for i in range(1, no_high_num+1, 1):
    #     video_name = no_high_prefix + str(i) + ".mp4"
    #     videos.append(video_name)

    # Talk Low
    for i in range(1, num_low_trim_talk+1, 1):
        video_name = low_trim_talk + str(i) + ".mp4"
        videos.append(video_name)

    # # No Low
    # for i in range(1, no_low_num + 1, 1):
    #     video_name = no_low_prefix + str(i) + ".mp4"
    #     videos.append(video_name)

    #
    # # Text High
    # for i in range(1, text_high_num+1, 1):
    #     video_name = text_high_prefix + str(i) + "_Trim.mp4"
    #     videos.append(video_name)

    # # Text Low
    # for i in range(1, text_low_num+1, 1):
    #     video_name = text_low_prefix + str(i) + "_Trim.mp4"
    #     videos.append(video_name)
    return videos


def pre_process(vid):
    video = cv2.VideoCapture(vid)

    fps = video.get(cv2.CAP_PROP_FPS)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    trim_video_list = []
    if duration > 0.5:
        times = np.arange(0, duration, 0.5)
        times = times.tolist()
    else:
        times = [0, duration]
    # Create temp folder to save the splitted videos in it
    trimmed_vid_path = os.path.join(os.getcwd(), "Temp\\")
    try:
        os.mkdir(trimmed_vid_path)
    except OSError as error:
        print(f"Directory {trimmed_vid_path} can not be created")
    # Split videos
    for start_time, end_time in zip(times[:-1], times[1:]):
        ffmpeg_extract_subclip(vid, start_time, end_time,
                               targetname=trimmed_vid_path + str(start_time) + ".mp4")
        trim_video_list.append(trimmed_vid_path + str(start_time) + ".mp4")

    return trim_video_list

def delete_temp_dir():
    # print(str(video))
    # detect = detection(labels, scores)
    # if detect:
    #     detection_list.append("detect")
    #     print(str(video) + ": DETECTION")
    # else:
    #     detection_list.append("not detect")
    trimmed_vid_path = os.path.join(os.getcwd(), "Temp\\")
    try:
        shutil.rmtree(trimmed_vid_path)
        print("Temp directory has deleted successfully")
    except OSError as error:
        print(f"Directory {trimmed_vid_path} can not be deleted")



def init_system():
    print("Init Start")
    # global avg_text
    global avg_look
    global avg_talk
    # avg_text = 0
    avg_look = 0
    avg_talk = 0
    # Choose to use a config and initialize the recognizer
    config = 'configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb.py'
    # Setup a checkpoint file to load
    checkpoint = 'checkpoints/tsn_r50_video_1x1x8_100e_kinetics600_rgb_20201015-4db3c461.pth'
    # Initialize the recognizer
    model = init_recognizer(config, checkpoint, device='cuda:0')
    # Use the recognizer to do inference

    label = 'tools/data/kinetics600/label_map_k600.txt'
    labels = open(label).readlines()
    labels = [x.strip() for x in labels]
    print("Init Done")
    return model, labels


def post_process(labels, results, video):
    threshold = 4
    results = [(labels[k[0]], k[1]) for k in results]
    detection = False
    # global avg_text
    global avg_look
    global avg_talk
    for result in results:
        # if result[0] == 'texting':
        #     if result[1] > 5.5:
        #         detection = True
        #     avg_text = avg_text + result[1]

        if result[0] == 'talking on cell phone':
            if result[1] > 2:
                detection = True
            avg_talk = avg_talk + result[1]

        if result[0] == 'looking at phone':
            if result[1] > 3:
                detection = True
            avg_talk = avg_talk + result[1]

        print(result)
        if detection:
            detection_list.append("detect")
            print(str(video) + ": DETECTION")
        else:
            detection_list.append("not detect")



def print_detections():
    for i, label in enumerate(detection_list):
        # if i == 0:
        #     print(" WITH PHONE LOW POSITION")
        # elif i == phone_low_num:
        #     print(" NO PHONE LOW POSITION")
        # elif i == (phone_high_num + phone_low_num):
        #     print(" WITH PHONE HIGH POSITION")
        # elif i == (phone_high_num + phone_low_num + no_high_num):
        #     print(" WITH PHONE HIGH POSITION")
        if label == "detect":
            print('video ' + str(i + 1) + ' DETECT')
        else:
            print('video ' + str(i + 1) + ' not detect')


def main():
    # our labels are 516 525 265
    videos = get_videos()
    model, labels = init_system()
    for video in videos:
        trim_video_list = pre_process(video)
        for trim_video in trim_video_list:
            # start_time = time.time()
            top5, top10, scores = inference_recognizer(model, trim_video)
            # end_time = time.time()
            # print(end_time - start_time)

            print(str(trim_video))
            post_process(labels, scores, trim_video)

        delete_temp_dir()


    print_detections()
    # print("average text " + str(avg_text/no_high_num))
    # print("average look " + str(avg_look/no_high_num))
    # print("average talk " + str(avg_talk/no_high_num))
    # print("average text " + str(avg_text/phone_high_num))
    # print("average look " + str(avg_look / no_high_num))
    # print("average talk " + str(avg_talk / 8))


main()
# split_video('demo/Epsh.mp4')
# split_video('demo/up/3.mp4')
# pre_process('demo/Epsh.mp4')
# post_process()