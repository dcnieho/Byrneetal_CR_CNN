# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:10:00 2019

@author: Marcus

Calibrate gaze data

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import helpers
import pathlib

plt.close('all')

#%% Parameters
plot_on = True

base_results_folder = r"results"
results_folders = ["ss01","ss02","ss03"]
video_base_folder = r"some/folder/to/CR dataset"
n_crs = 1

#  Info required to convert to deg
scr_res             = [1920, 1080]
scr_size            = [531, 299]
scr_viewDist        = 790
scr_FOV             = 2*np.arctan(np.array(scr_size)/2./scr_viewDist)

deg2PixFunc         = lambda x: np.mean(np.tan(x/2)/np.tan(scr_FOV/2)*scr_res)
pix2DegFunc         = lambda x: np.mean(2*np.arctan(x*np.tan(scr_FOV/2)/scr_res))*180/np.pi

for v in results_folders:
    results_folder = str(pathlib.Path(base_results_folder) / v)
    video_folder = str(pathlib.Path(video_base_folder) / v)
    # File that contains pupil and cr locations extracted from eye images
    data_files = pathlib.Path(results_folder).glob('cam1_*.tsv')
    df_feature_data = pd.concat((pd.read_csv(rec, delimiter='\t').assign(point_number=i+1) for i,rec in enumerate(data_files)), ignore_index=True)

    # Use P-CR
    feature_x = 'pup_x'
    feature_y = 'pup_y'

    # measurement trials
    calibration_trials = np.arange(1, 10) # Calibration trials 1-9
    measurement_trials = np.arange(10, df_feature_data.point_number.max()+1)
    all_trials = np.hstack((calibration_trials, measurement_trials))

    # Calibration grid used
    tar_df = pd.read_csv(pathlib.Path(video_folder)/'targetPos_1stFrame_pixels.txt', delimiter='\t',header=None, names=['tar_x', 'tar_y'])
    tar_df['tar_x_deg'] = tar_df.apply(lambda row: pix2DegFunc(row['tar_x']-scr_res[0]/2), axis=1)
    tar_df['tar_y_deg'] = tar_df.apply(lambda row: pix2DegFunc(row['tar_y']-scr_res[1]/2), axis=1)

    xy_target = tar_df.iloc[0:9,2:4].to_numpy()


    # %%
    # Plot feature data for the calibration points
    plt.figure()
    df_temp = df_feature_data[[x in calibration_trials for x in df_feature_data.point_number]]
    plt.plot(df_temp.pup_x - df_temp.cr1_x_thresh, df_temp.pup_y - df_temp.cr1_y_thresh, '*')
    plt.plot(df_temp.pup_x - df_temp.cr1_x_CNN, df_temp.pup_y - df_temp.cr1_y_CNN, '*')
    plt.plot(df_temp.pup_x - df_temp.cr1_x_rad_symm, df_temp.pup_y - df_temp.cr1_y_rad_symm, '*')


    # Plot calibration targets
    plt.plot(xy_target[:, 0], xy_target[:, 1], '*')

    plt.legend(['P-CR thresh (pixels)', 'P-CR CNN (pixels)', 'P-CR rad symm (pixels)', 'cal targets (deg)'])
    plt.show()


    # calibrate pupil-CR for the three CR features
    cal_df = [[],[],[]]
    for i,feat in enumerate(['thresh','CNN','rad_symm']):

        crx = 'cr1_x_'+feat
        cry = 'cr1_y_'+feat

        xy_raw = []

        for p in calibration_trials:
            df_point = df_feature_data[df_feature_data.point_number == p]

            pcr_x = df_point[feature_x] - df_point[crx]
            pcr_y = df_point[feature_y] - df_point[cry]
            xy_raw.append([np.nanmedian(pcr_x), np.nanmedian(pcr_y)])


        #%% Find mapping function
        eye_data   = np.array(xy_raw)
        cal_targets= xy_target[calibration_trials-1,:]

        gamma_x, _ = helpers.biqubic_calibration_with_cross_term(eye_data[:, 0], eye_data[:, 1],
                                                                              cal_targets[:, 0])
        gamma_y, _ = helpers.biqubic_calibration_with_cross_term(eye_data[:, 0], eye_data[:, 1],
                                                                              cal_targets[:, 1])


        #%% Apply mapping function
        for p in all_trials:
            # Read json file with timestamps to find timestamps and to decide
            # which frames to be included
            pid_str = f'{p:03d}'
            json_path = pathlib.Path(video_folder) / ('cam1_R' + pid_str + '_info+.json')

            with open(json_path) as f:
                data = json.load(f)

            ts = []
            to_analyze = []
            for d in data:
                ts.append(d['systemTimeStamp'] * 1000)  # to ms
                to_analyze.append(d['toAnalyze'])

            # Remove some frames (before and after the actual recording started)
            df_ts = pd.DataFrame(ts)
            df_ts = df_ts[to_analyze]
            df_ts.reset_index(inplace=True)
            df_ts.drop('index', inplace=True, axis=1)

            df_ts.columns = ['time']
            df_ts['frame_no'] = np.array(df_ts.index) + 1

            df_temp = df_feature_data[df_feature_data.point_number == p]
            df_temp = df_temp[to_analyze]
            df_temp.reset_index(inplace=True)
            df_temp.drop('index', inplace=True, axis=1)
            df_temp['frame_no'] = np.array(df_temp.index) + 1

            if len(df_temp) == 0:
                continue

            # compute p-cr, cr, and pupil data
            cr_x = df_temp[crx]
            cr_y = df_temp[cry]
            pupil_x = df_temp[feature_x]
            pupil_y = df_temp[feature_y]
            pcr_x = pupil_x - cr_x
            pcr_y = pupil_y - cr_y

            frame_number = np.array(df_temp.frame_no).astype('int')

            # Estimate gaze from pcr
            xy = np.vstack((pcr_x, pcr_y)).T

            g_x = helpers.biqubic_estimation_with_cross_term(xy[:,0], xy[:,1], gamma_x)
            g_y = helpers.biqubic_estimation_with_cross_term(xy[:,0], xy[:,1], gamma_y)

            # Make dataframe and save to file
            D = np.c_[np.repeat(p, len(g_x)), frame_number, np.vstack((g_x, g_y)).T]

            df_calibrated = pd.DataFrame(data=D, columns=['trial', 'frame_no', 'x_'+feat, 'y_'+feat])

            df_cal = pd.merge(df_ts, df_calibrated, on="frame_no")

            cal_df[i].append(df_cal)


    # store calibrated data to file
    df_temp = [pd.concat(x) for x in cal_df]
    df_out = df_temp[0]
    for i in range(1,len(df_temp)):
        df_out = pd.merge(df_out,df_temp[i].drop('time',axis=1), on=['trial','frame_no'])
    df_out.to_csv(pathlib.Path(results_folder)/'calibrated_data.tsv', sep='\t', float_format='%.8f', na_rep='nan',index=False)
