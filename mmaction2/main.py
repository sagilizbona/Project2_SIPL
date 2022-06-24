
from mmaction.apis import inference_recognizer, init_recognizer
import time
import cv2

def init_system():
    print("Init Start")
    # Choose to use a config and initialize the recognizer
    config = 'configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb.py'
    # config = 'configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py'
    # Setup a checkpoint file to load
    checkpoint = 'checkpoints/tsn_r50_video_1x1x8_100e_kinetics600_rgb_20201015-4db3c461.pth'
    # checkpoint = 'checkpoints/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth'
    # Initialize the recognizer
    model = init_recognizer(config, checkpoint, device='cuda:0')
    # Use the recognizer to do inference
    # video = ['demo/demo.mp4','demo/Walk.mp4','demo/Walk1.mp4']
    # video = ['demo/Epsh1.mp4']
    # video = ['demo/front/1.mp4','demo/front/2.mp4','demo/front/3.mp4']
    # video = ['demo/up/1.mp4','demo/up/2.mp4','demo/up/3.mp4','demo/up/4.mp4','demo/up/5.mp4','demo/up/6.mp4','demo/up/7.mp4','demo/up/8.mp4','demo/up/9.mp4','demo/up/10.mp4']
    # video = ['demo/up_short/1.mp4','demo/up_short/2.mp4','demo/up_short/3.mp4','demo/up_short/4.mp4','demo/up_short/5.mp4']
    # video = ['demo/front_side/1.mp4','demo/front_side/2.mp4','demo/front_side/3.mp4','demo/front_side/4.mp4']
    video = ['demo/splitter.mp4','demo/splitter2.mp4','demo/splitter3.mp4', 'demo/2.mp4']
    label = 'tools/data/kinetics600/label_map_k600.txt'
    labels = open(label).readlines()
    labels = [x.strip() for x in labels]
    print("Init Done")
    return model, video, labels


def detection(labels, results):

    threshold = 4
    results = [(labels[k[0]], k[1]) for k in results]
    detection = False
    for result in results:
    #     # print(f'{result[0]}: ', result[1])
    #     if result[0] == 'looking at phone':
    #         detection = True
    #     elif result[0] == 'texting':
    #         detection = True
    #     elif result[0] == 'talking on cell phone':
    #         detection = True

        print(result)
        if result[1] > threshold:
            detection = True
    return detection


def run_interval(model, video, labels):
    print("Run interval")
    start_time = time.time()
    top5, top10, scores = inference_recognizer(model, video)
    end_time = time.time()
    print(end_time - start_time)
    print( str(video))
    detect = detection(labels, scores)

    if detect:
        print(str(video) + ": DETECTION")




def main():
    # our labels are 516 525 265
    model, video, labels = init_system()
    for vid in video:
        cap = cv2.VideoCapture(vid)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        run_interval(model, vid, labels)



def split_video(required_video_file):

    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

    # # Replace the filename below.
    # required_video_file = "filename.mp4"

    with open("demo/times.txt") as f:
      times = f.readlines()

    times = [x.strip() for x in times]

    for time in times:
      starttime = int(time.split("-")[0])
      endtime = int(time.split("-")[1])
      ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname="demo/"+str(times.index(time)+1)+".mp4")


def splitter():
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    ffmpeg_extract_subclip("demo/2.mp4", 0.6, 1, targetname="demo/splitter2.mp4")

main()
# split_video('demo/2.mp4')
# split_video('demo/up/3.mp4')
# splitter()