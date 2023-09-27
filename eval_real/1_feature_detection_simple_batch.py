# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:27:33 2019

@author: Marcus
"""

import cv2
import numpy as np
import time
import sys
from matplotlib import pyplot as plt
import pathlib
import pandas as pd
import helpers
import warnings


n_crs = 1
IMAGE_SIZE = (180,180)
plt.close('all')

model = 'stage_2'
video_base_folder = r"some/folder/to/CR dataset"
video_folders = ["ss01","ss02","ss03"]

def do_shit(video_file, result_folder):
    from tensorflow import keras
    from deeptrack.extras.radialcenter import radialcenter

    param_file = pathlib.Path(video_file).parent/'cam1_params.tsv'

    # Create list with header names or relevant CRs
    # both binary and intensity based output
    cr_list = []
    for k in range(n_crs):
        cr_list.append("cr{}_x_thresh".format(k + 1))
        cr_list.append("cr{}_y_thresh".format(k + 1))
        cr_list.append("cr{}_area".format(k + 1))
        cr_list.append("cr{}_x_CNN".format(k + 1))
        cr_list.append("cr{}_y_CNN".format(k + 1))
        cr_list.append("cr{}_x_rad_symm".format(k + 1))
        cr_list.append("cr{}_y_rad_symm".format(k + 1))


    try:
        df_params = pd.read_csv(param_file, sep='\t')
    except:
        print('Error: Run 0_threshold_estimation first!')
        sys.exit()

    #%% Detection parameters (in relatation to iris diameter and intensity
    # threshold of the pupil, cr, and iris)

    iris_area = int(np.pi * (df_params['iris_diameter'].iloc[0] / 2) **2 )
    pupil_area_min = int(np.pi * (df_params['iris_diameter'].iloc[0] / 16) **2 )
    cr_area_min = 2
    cr_area_max = pupil_area_min
    cr_size_limits = [cr_area_min, cr_area_max]
    batch_size=100


    pupil_size_limits = [pupil_area_min, iris_area] # min, max in pixels

    pupil_intensity_threshold = int(df_params['pup_threshold'].iloc[0])
    cr_intensity_threshold = int(df_params['cr_threshold'].iloc[0])

    pupil_cr_distance_max = int(df_params['iris_diameter'].iloc[0] / 4)

    test_model = keras.models.load_model(f'..\\trained_model\\{model}.h5', compile=False)

    out_data_file= pathlib.Path(result_folder) / (video_file.stem + '.tsv')



    #%% Read and threshold image to find pupil

    data_out = []


    cap = cv2.VideoCapture(str(video_file))
    total_nr_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cal_point_number = 1
    print(str(video_file), total_nr_of_frames, cal_point_number)


    k = 0
    # %%
    done= False
    t0 = time.time()
    while not done:
        # Capture frame-by-frame
        to_process = {}
        while len(to_process)<batch_size:
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

            if frame is None:
                print(frame_number, k, total_nr_of_frames)
                done = True
                break

            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Extract ROI in eye image
            img = img[int(df_params['y_ul'].iloc[0]):int(df_params['y_lr'].iloc[0]),
                      int(df_params['x_ul'].iloc[0]):int(df_params['x_lr'].iloc[0])]



            # %% Detect pupil
            # (cx, cy), area, countour_points, ellipse
            pupil_features = helpers.detect_pupil(img,
                                            pupil_intensity_threshold,
                                            pupil_size_limits,
                                            window_name=None)

            if np.isnan(pupil_features['area']):
                print(f'{int(frame_number)}: pupil not found ({video_file.name})')

            # %% Detect CR
            cr_features, cr_patch, patch_off, cr_cnts = helpers.detect_cr(img, cr_intensity_threshold,
                                                cr_size_limits,
                                                pupil_cr_distance_max, pupil_features['cog'],
                                                no_cr = n_crs,
                                                cr_img_size = IMAGE_SIZE,
                                                window_name=None)

            if cr_patch is not None:
                cr_patch,_ = cr_patch
                # black out everything well beyond CR
                cr_local = [int(x-y) for x,y in zip(cr_features[0][0:2],patch_off[0])]
                radius = 48 # about 3 times CR horizontal diameter

                mask = np.zeros_like(cr_patch)
                mask = cv2.circle(mask, cr_local, radius, (255,255,255), -1)
                cr_patch2 = cv2.bitwise_and(cr_patch, mask).astype(np.float64)/255
            else:
                cr_patch2 = None

            to_process[frame_number] = (pupil_features, cr_features, cr_patch2, patch_off)

        print(f'{video_file.name}: {int(frame_number)}/{total_nr_of_frames}')


        frames = [(fr,to_process[fr][2]) for fr in to_process if to_process[fr][2] is not None]
        if frames:
            measured_positions = test_model.predict(np.array([fr[1] for fr in frames]), verbose=0)*IMAGE_SIZE

        for frame_number in to_process:
            pupil_features, cr_features, cr_patch2, patch_off = to_process[frame_number]

            if cr_patch2 is None:
                measured_position_x, measured_position_y = np.nan, np.nan
                radial_x, radial_y = np.nan, np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore") # ignore warnings from radialcenter
                    radial_x, radial_y = radialcenter(cr_patch2)
                    radial_x += patch_off[0][0]
                    radial_y += patch_off[0][1]

                for i,data in enumerate(frames):
                    if data[0]==frame_number:
                        measured_position_x = measured_positions[i,0]+patch_off[0][0]
                        measured_position_y = measured_positions[i,1]+patch_off[0][1]
                        break
            if len(cr_features)==0:
                cr_features = [np.array([np.nan,np.nan,np.nan])]

            temp_list = [str(video_file), frame_number,
                            pupil_features['cog'][0],
                            pupil_features['cog'][1],
                            pupil_features['area']] + \
                            [x for b in cr_features for x in b] + \
                            [measured_position_x, measured_position_y] + \
                            [radial_x, radial_y] + \
                            [cal_point_number]

            data_out.append(temp_list)

        k += 1

    # pp.close()
    print('fps: {}'.format(total_nr_of_frames / (time.time() - t0)))
    print(len(data_out))

    # %% Save data to dataframe
    df = pd.DataFrame(data_out, columns = ['video_name',
                                           'frame_no',
                                           'pup_x','pup_y', 'pup_area'] + \
                                            cr_list + \
                                           ['point_number'])


    # Save files if not in debug mode
    df.to_csv(out_data_file, sep='\t', float_format='%.8f', na_rep='nan',index=False)

    # When everything done, release the capture
    cap.release()

    return video_file



if __name__ == "__main__":
    import pebble

    with pebble.ProcessPool(max_workers=2, max_tasks=1) as pool:
        for v in video_folders:
            video_folder = pathlib.Path(video_base_folder) / v
            results_folder = f"results\\{pathlib.Path(video_folder).name}"

            if not pathlib.Path(results_folder).is_dir():
                pathlib.Path(results_folder).mkdir(parents=True)


            video_files = list(pathlib.Path(video_folder).glob('*.mp4'))
            res_folds   = [results_folder]*len(video_files)
            for result in pool.map(do_shit, video_files, res_folds).result():
                print(f'done with {result}')
