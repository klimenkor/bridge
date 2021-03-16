import os, glob

# live
path_to_new = "/home/nvr/ftp/new"
path_to_frames = "/home/nvr/ftp/frames"
path_to_archive = "/home/nvr/ftp/archive"
path_to_model = "/home/pi/bridge/ssdlite_mobilenet_v2_coco_2018_05_09"
path_to_data = "/home/pi/bridge/data"

# dev

# path_to_new = "/Users/rklimenko/dev/bridge/media/new"
# path_to_frames = "/Users/rklimenko/dev/bridge/media/frames"
# path_to_archive = "/Users/rklimenko/dev/bridge/media/archive"
# path_to_model = "/Users/rklimenko/dev/bridge/ssdlite_mobilenet_v2_coco_2018_05_09"
# path_to_data = "/Users/rklimenko/dev/bridge/data"

# proto_txt = "MobileNetSSD/MobileNetSSD_deploy.prototxt"
# model = "MobileNetSSD/MobileNetSSD_deploy.caffemodel"

def file_is_ready(file):
    if os.path.exists(file):
        try:
            os.rename(file, file)
            return True
        except OSError as e:
            return False

def get_ordered_files(path):
    files = glob.glob(path + '/*.*')
    files.sort(key=os.path.getmtime)
    return files