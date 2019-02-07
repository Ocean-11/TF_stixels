

#from __future__ import absolute_import


import os
import glob

if __name__ == '__main__':

    create_video = False

    image_width = 476  # when image width = 480
    in_folder_name = '/media/vision/Results/image_for_predict'
    #in_folder_name = '/media/vision/Results/test_video_GC23_1'
    #in_folder_name = '/media/vision/Results/test_video_GC23_2'
    #in_folder_name = 'test_video_GC23_2'
    #in_folder_name = 'test_video_Site40_1'
    #in_folder_name = 'test_video_single'
    #in_folder_name = 'test_video_NLSite_1'

    '''
    in_folder_name = 'test_video_UKSite4GC'
    image_width = 636  # when image width = 636
    '''
    '''
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    model_dir = filedialog.askdirectory(initialdir='/home/dev/PycharmProjects/stixel/TF_stixels/results')
    root.destroy()
    '''

    images = sorted(glob.glob(in_folder_name + '/*.jpg'))

    out_folder_name = in_folder_name

    # Determine the model to be used for inference
    model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-01-28_18-57-33_EP_250'
    model_name = os.path.basename(model_dir)


    import image_predict

    #image_in = '/media/vision/Results/image_for_predict/frame_000142.jpg'
    #out_folder_name = os.path.dirname(image_in)

    out_folder_name = in_folder_name


    for image in images:
        image_predict.main(image, out_folder_name, model_dir, image_width, False, show_images=False)


    if create_video:
        import video

        video.main(out_folder_name, 'video.avi')


