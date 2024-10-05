import subprocess
import pandas as pd
import os
import ctypes
from scipy.ndimage import gaussian_filter1d
import numpy as np
from pykalman import KalmanFilter

JOINTS_NAME = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
               'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
               'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
               'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
               'Knee.L_x', 'Knee.L_y', 'Knee.L_z',
               'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
               'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z',
               'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z',
               'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z',
               'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
               'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
               'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z',
               'Neck_x', 'Neck_y', 'Neck_z',
               'Head_x', 'Head_y', 'Head_z',
               'Nose_x', 'Nose_y', 'Nose_z',
               'Eye.L_x', 'Eye.L_y', 'Eye.L_z',
               'Eye.R_x', 'Eye.R_y', 'Eye.R_z',
               'Ear.L_x', 'Ear.L_y', 'Ear.L_z',
               'Ear.R_x', 'Ear.R_y', 'Ear.R_z']

class BVHExporter:
    def __init__(self):
        self.df_list = []
        self.kalman_filter = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        self.gaussian_sigma = 3.0  

    def add(self, joints3d):
        joints_info = pd.DataFrame(joints3d.reshape(1, 57), columns=JOINTS_NAME)

        # Invert y and z axes to match the BVH coordinate system
        joints_info.iloc[:, 1::3] = joints_info.iloc[:, 1::3] * -1
        joints_info.iloc[:, 2::3] = joints_info.iloc[:, 2::3] * -1

        # Calculate the hip center position
        hipCenter = joints_info.loc[:, ['Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                                        'Hip.L_x', 'Hip.L_y', 'Hip.L_z']]

        joints_info['hip.Center_x'] = hipCenter.iloc[0][::3].sum() / 2
        joints_info['hip.Center_y'] = hipCenter.iloc[0][1::3].sum() / 2
        joints_info['hip.Center_z'] = hipCenter.iloc[0][2::3].sum() / 2

        self.df_list.append(joints_info.copy())

    def smooth_data(self):
        all_data = pd.concat(self.df_list, ignore_index=True)

        smoothed_data = all_data.copy()

        for col in JOINTS_NAME:
            if col in smoothed_data.columns:
                kalman_smoothed, _ = self.kalman_filter.smooth(smoothed_data[col].values)
                smoothed_data[col] = kalman_smoothed
                smoothed_data[col] = gaussian_filter1d(smoothed_data[col], sigma=self.gaussian_sigma)

                ema_span = 10 
                smoothed_data[col] = smoothed_data[col].ewm(span=ema_span, adjust=False).mean()

                def low_pass_filter(data, cutoff=0.1):
                    from scipy.signal import butter, filtfilt
                    b, a = butter(4, cutoff, btype='low', analog=False)
                    return filtfilt(b, a, data)

                smoothed_data[col] = low_pass_filter(smoothed_data[col])

                smoothed_data[col] = smoothed_data[col].interpolate()

        def spline_interpolation(df, column_name, smooth_factor=0.1):
            from scipy.interpolate import UnivariateSpline
            x = np.arange(len(df))
            y = df[column_name].values
            spline = UnivariateSpline(x, y, s=smooth_factor)
            return spline(x)

        for col in JOINTS_NAME:
            if col in smoothed_data.columns:
                smoothed_data[col] = spline_interpolation(smoothed_data, col)

        return smoothed_data


    #--------------- remove the below code for leg tracking --------------------------------------------------
    def fix_leg_position(self, bvh_data):
        # Identify the joints for legs and thighs
        leg_joints = ['Hip.L_x', 'Hip.L_y', 'Hip.L_z',
                    'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                    'Knee.L_x', 'Knee.L_y', 'Knee.L_z',
                    'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
                    'Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z']

        for joint in leg_joints:
            if joint in bvh_data.columns:
                kalman_smoothed, _ = self.kalman_filter.smooth(bvh_data[joint].values)
                bvh_data[joint] = kalman_smoothed
                bvh_data[joint] = gaussian_filter1d(bvh_data[joint], sigma=8.0)
        
        for joint in leg_joints:
            if joint in bvh_data.columns:
                mean_position = bvh_data[joint].mean()
                bvh_data[joint] = mean_position

        return bvh_data
    
    #--------------- remove the above code for leg tracking --------------------------------------------------



    def dump(self, bvh_path):
        smoothed_data = self.smooth_data()

    #--------------- remove the below code for leg tracking --------------------------------------------------
    
        fixed_data = self.fix_leg_position(smoothed_data)
    
    #--------------- remove the above code for leg tracking --------------------------------------------------
        
        if not os.path.isdir('result'):
            os.mkdir('result')

        csv_path = bvh_path.replace('.bvh', '.csv')
        fixed_data.to_csv(csv_path, index=False)

        proc = subprocess.Popen(['blender', '--background', 'blender_scripts/csv_to_bvh.blend', '-noaudio', '-P', 'blender_scripts/csv_to_bvh.py', '--', csv_path, bvh_path], shell=True)
        proc.wait()
        (stdout, stderr) = proc.communicate()

        if proc.returncode != 0:
            ctypes.windll.user32.MessageBoxW(0, u"Blender not found, install Blender and add to your PATH first.", u"Error", 0)
        else:
            print("[INFO] Success.")
