import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os,sys
import glob
import threading
import time
from os import rename, remove
import cv2

import ikncu

class Detector:
    
    private = 0

    def __init__(self):
        self.private = 1


    def process(self,frame):
        print("...processing frame %d" % (self.private))


class Watcher:
    DIRECTORY_TO_WATCH = ikncu.path_to_new

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()

class Splitter:

    @staticmethod
    def process(path): 
        
        #wait for upload to finish
        size=os.path.getsize(path)
        same=0
        while same < 3:
            time.sleep(1)

            if size == os.path.getsize(path):
                same = same + 1
            else:
                same = same - 1
                size = os.path.getsize(path)

        print("     %s: start processing" % (path))

        try:
            file = os.path.basename(path)

            sep = '.'

            video = cv2.VideoCapture(path)
            fps = int(video.get(cv2.CAP_PROP_FPS))
            print("   %6.2f fps" % fps)

            # capture 1 frame per second
            count = 0
            start = time.process_time()
            success, frame = video.read()
            while success:
                success, frame = video.read()
                if not success:
                    print("   cannot read!")
                    break

                if count % fps == 0:
                    frame_file = "%s-%#05d.jpg" % (os.path.join(ikncu.path_to_frames, file.split(sep, 1)[0]), count + 1)
                    
                    ######## detect an object
                    ######
                    ######

                    detector.process(frame)
                    
                    print("    saving to %s" % (frame_file))
                    cv2.imwrite(frame_file, frame)

                count = count + 1

            video.release()
            end = time.process_time() - start
            if fps>0:
                print("    % d frames in %.2f seconds" % (count / fps, end - start))
            else:
                print("    0 fps")
        
        except:
            print("[processVideos] Unexpected error:", sys.exc_info()[0])
            processes.remove(path) 
            raise    
            #remove(path)

        try:
            rename(path, os.path.join(ikncu.path_to_archive, file))
        except:
            print("Warning: file already exists")
         
        #processes.remove(path)    
        return True

class Handler(FileSystemEventHandler):

    @staticmethod
    def file_is_ready(file):
        if os.path.exists(file):
            try:
                os.rename(file, file)
                return True
            except OSError as e:
                return False

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif (event.event_type == 'created') and event.src_path not in processes:
            processes.append(event.src_path)
            print("%d: %s - added to queue" % (len(processes),event.src_path))

            threading.Thread(target=Splitter.process, args=(event.src_path,)).start()


if __name__ == '__main__':
    processes = []
    detector = Detector()
    w = Watcher()
    w.run()