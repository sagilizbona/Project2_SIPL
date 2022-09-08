
import torch
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger


###############################################################################
# commands to download video from youtube and trim it
# conda install youtube-dl
# !youtube-dl https://www.youtube.com/watch?v=R1SQcXhqefs -f 22 -o video1.mp4
# !ffmpeg -i video1.mp4 -t 00:00:06 -c:v copy video-clip1.mp4
###############################################################################


def ExtractVideo2Frames():
    # Make sure that the video is in the project directory
    # vidcap = cv2.VideoCapture('video-clip1.mp4')
    vidcap = cv2.VideoCapture('1.mp4')


    # get the frames from the video

    success,image = vidcap.read()
    number_of_frames = 0
    # print(success)
    while success:
      cv2.imwrite("new_folder/frame%d.jpg" % number_of_frames, image)     # save frame as JPEG file
      success,image = vidcap.read()
      number_of_frames += 1

    #optional view of video frame
    im = cv2.imread("./new_folder/frame1.jpg")
    # cv2_imshow(im)
    # cv2.imshow("pic",im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(number_of_frames)
    return number_of_frames


# Inference with a keypoint detection model
def GetPredictor():
    cfg = get_cfg()   # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def CreateNewFrames(number_of_frames, predictor_function, cfg):
    iterator = 0
    img_array = []
    print(number_of_frames)
    while iterator != number_of_frames:
        im = cv2.imread("./new_folder/frame%d.jpg" % iterator)
        outputs = predictor_function(im)
        # final_output += outputs
    # im = cv2.imread("./frame1.jpg")
    # outputs = predictor(im)
        v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img_array.append(out.get_image()[:, :, ::-1])
        cv2.imwrite("./new_folder/new_frame%d.jpg" % iterator, out.get_image()[:, :, ::-1])     # save new frame as JPEG file
        iterator += 1
    return out


# function to create new list of frames in order for making a new video
def special_sort(size):
    i = 0
    sorted_list = []
    while i != size:
        sorted_list.insert(len(sorted_list), "./new_folder/new_frame%d.jpg" %i)
        i += 1
    return sorted_list


def CreateVideoFromFrames(number_of_frames,image_size):

    # Get new frames in list and create video from the list
    print("create new video")
    video_name = 'proj.mp4'
    images = special_sort(number_of_frames)
    height, width, layers = image_size.get_image()[:, :, ::-1].shape
    size = (width,height)
    out_new_vid = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
    print("working on the array")
    image_array = []
    for filename in images:
        imge = cv2.imread(os.path.join(filename))
        image_array.append(imge)
        out_new_vid.write(imge)
    print("start to release")
    out_new_vid.release()


if __name__ =="__main__":
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

    setup_logger()

    number_of_frames = ExtractVideo2Frames()
    predictor, cfg = GetPredictor()
    image_size = CreateNewFrames(number_of_frames,predictor,cfg)
    CreateVideoFromFrames(number_of_frames,image_size)
