

#from __future__ import absolute_import


import os
import glob
import sys

if __name__ == '__main__':

    create_video = False

    image_width = 476  # when image width = 480
    #in_folder_name = '/media/vision/Results/image_for_predict'
    #in_folder_name = '/media/vision/Results/test_video_GC23_1'
    #in_folder_name = '/media/vision/Results/test_video_GC23_1_BW'
    #in_folder_name = '/media/vision/Results/test_video_GC23_2'
    #in_folder_name = '/media/vision/Results/test_video_Site40_1'
    #in_folder_name = '/media/vision/Results/test_video_BW'
    #in_folder_name = 'test_video_GC23_2'
    #in_folder_name = 'test_video_Site40_1'
    #in_folder_name = 'test_video_single'in_folder_name = '/media/vision/Results/test_video_Site40_1'
    #in_folder_name = 'test_video_NLSite_1'


    image_width = 551 # image width = 555
    #in_folder_name = '/media/dnn/ML/Results/test_video_session1_G'
    #in_folder_name = '/media/vision/Results/for_galit'
    #in_folder_name = '/media/vision/Results/test_video_sessionA'
    in_folder_name = '/media/dnn/ML/Results/test_video_sessionA_1'

    #in_folder_name = '/media/dnn/ML/Results/test_video_UKSite4GC'
    #image_width = 636  # when image width = 636


    '''
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    model_dir = filedialog.askdirectory(initialdir='/home/dev/PycharmProjects/stixel/TF_stixels/results')
    root.destroy()
    '''
    images = sorted(glob.glob(in_folder_name + '/*.jpg'))

    out_folder_name = in_folder_name

    # Determine the model to be used for inference
    model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-04-10_18-44-40_EP_250'
    #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-03-07_18-36-03_EP_250'
    #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-01-28_18-57-33_EP_250'
    #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-02-19_18-40-42_EP_250'
    #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-02-24_09-20-44_EP_250'
    #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-02-10_17-43-39_EP_250' # BW model!!
    model_name = os.path.basename(model_dir)

    os.chdir(model_dir)
    sys.path.insert(0, os.getcwd())

    image_out_dir = in_folder_name + '/' + model_name
    if not os.path.exists(image_out_dir) and not os.path.isdir(image_out_dir):
        os.mkdir(image_out_dir)

    '''
    for image in images:
        image_predict.main(image, out_folder_name, model_dir, image_width, False, show_images=False)
        '''

    # Create image_predictor object
    from image_predict import image_predictor
    predictor = image_predictor(images[0], image_out_dir, image_width, model_dir, debug_image=True, show_images=False)

    # Run through the images and create predictions & visulizations
    for image in images:
        #image_predict.main(image, out_folder_name, model_dir, image_width, False, show_images=False)
        predictor.predict(image)

    # Close the session
    predictor.close_session()

    if create_video:
        import video

        video.main(out_folder_name, 'video.avi')



