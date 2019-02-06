
import os
import cv2
import tkinter as tk
from tkinter import filedialog
import glob

def main(images_folder, video_name):

    images = sorted(glob.glob(images_folder + '/*.jpg'))
    # images = [img for img in os.listdir((images_folder)) if img.endswith('.jpg')]

    # Filter out debug images
    clean_images = [k for k in images if 'debug' not in k]

    frame_name = os.path.join(images_folder, images[0])
    #print(frame_name)
    frame = cv2.imread(frame_name)
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video = cv2.VideoWriter(video_name, fourcc, 2, (width, height))

    for image in clean_images:
        video.write(cv2.imread(os.path.join(images_folder, image)))
        print('insert {}'.format(image))

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':

    images_folder = '/media/vision/Results/image_for_predict/'

    ' when executed as a script, open a GUI window to select the presented TFrecord file '
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    images_folder = filedialog.askdirectory(initialdir='/media/vision/Results/image_for_predict/')
    root.destroy()

    video_name = os.path.join(images_folder, 'video.avi')
    main(images_folder, video_name)








