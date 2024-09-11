import os
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import butter, filtfilt
import yaml
import time
import sys
from scipy.interpolate import interp1d
"""
Load Parameters
"""

# Load configuration parameters from a YAML file
with open('paras copy.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Define the current directory containing data
current_dir = "/scai_data2/scai_datasets/interim/scai-outsense/"

# Extract various parameters from the loaded YAML configuration
seed_number = params['seed_number']
raw_data_pars = params['raw_data_pars']
upsample_freq = params['upsample_freq']
downsample_freq = params['downsample_freq']
filter_parameters = params['filter_parameters']

# Extract specific filter parameters
lowcut_kinematic = filter_parameters['lowcut_kinematic']
highcut_kinematic = filter_parameters['highcut_kinematic']
filter_order = filter_parameters['filter_order']

# Define data loading function without directory traversal
def data_loader_no_dir(subject_dir, modality, settings):
    modality_dir = os.path.join(subject_dir, modality)
    if os.path.exists(modality_dir):
        data = []
        time_ranges = []  # Store time ranges for each file
        files = [f for f in os.listdir(modality_dir) if f.endswith(tuple(settings['file_format']))]
        for file_name in files:
            file_path = os.path.join(modality_dir, file_name)
            if os.path.getsize(file_path) > 0:
                df = pd.read_csv(file_path, compression='gzip')
                if not df.empty:
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    start_time = df['time'].min()
                    end_time = df['time'].max()
                    time_ranges.append((file_name, start_time, end_time))
                    data.append(df)
        if data:
            df_modality = pd.concat(data).reset_index(drop=True)
            df_modality = df_modality.sort_values(by=['time']).reset_index(drop=True)
            df_modality = df_modality.set_index('time')
            df_modality = df_modality[settings['data_columns']]
            return df_modality, time_ranges
    return pd.DataFrame(), []

# Define data loading function with directory traversal (similar to the above but includes directory traversal)
def data_loader_with_dir(subject_dir, modality, settings):
    """
    Load data files from a specified directory and modality, including subdirectories.
    """
    modality_dir = os.path.join(subject_dir, modality)
    data = []
    time_ranges = []  # List to store time ranges for each file
    
    if 'dirs' in settings['file_format']:
        for root, dirs, files in os.walk(modality_dir):
            for file_name in files:
                if file_name.endswith('.gz'):
                    file_path = os.path.join(root, file_name)
                    process_file(file_path, data, time_ranges)
    else:
        files = [f for f in os.listdir(modality_dir) if f.endswith(tuple(settings['file_format']))]
        for file_name in files:
            file_path = os.path.join(modality_dir, file_name)
            process_file(file_path, data, time_ranges)

    if data:
        df_modality = pd.concat(data).reset_index(drop=True)
        df_modality['time'] = pd.to_datetime(df_modality['time'], unit='s')
        df_modality = df_modality.sort_values(by=['time']).reset_index(drop=True)
        df_modality = df_modality.set_index('time')
        df_modality = df_modality[settings['data_columns']]
        return df_modality, time_ranges
    return pd.DataFrame(), []

def process_file(file_path, data, time_ranges):
    """
    Helper function to process individual files, checking size and loading contents.
    """
    if os.path.getsize(file_path) > 0:
        df = pd.read_csv(file_path, compression='gzip')
        if not df.empty:
            # Assuming 'time' is in seconds since epoch
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # Append time range information
            start_time = df['time'].min()
            end_time = df['time'].max()
            time_ranges.append((os.path.basename(file_path), start_time, end_time))  # Using basename for file name
            data.append(df)

# Data loader function for reading sensor data with timestamp in milliseconds
def data_loader_no_dir_ms(subject_dir, modality, settings):
    modality_dir = os.path.join(subject_dir, modality)
    if os.path.exists(modality_dir):
        data = []
        time_ranges = []
        files = [f for f in os.listdir(modality_dir) if f.endswith(tuple(settings['file_format']))]
        for file_name in files:
            file_path = os.path.join(modality_dir, file_name)
            if os.path.getsize(file_path) > 0:
                df = pd.read_csv(file_path, compression='gzip')
                if not df.empty:
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    start_time = df['time'].min()
                    end_time = df['time'].max()
                    time_ranges.append((file_name, start_time, end_time))
                    data.append(df)
        if data:
            df_modality = pd.concat(data).reset_index(drop=True)
            df_modality = df_modality.sort_values(by=['time']).reset_index(drop=True)
            df_modality = df_modality.set_index('time')
            df_modality = df_modality[settings['data_columns']]
            return df_modality, time_ranges
    return pd.DataFrame(), []

# Process modality data, checking and adjusting for repeated timestamps
def process_modality(data_df, sample_rate):
    if data_df.empty:
        print("No data to process.")
        return data_df

    print("Checking for repeated timestamps...")
    original_count = data_df.index.duplicated().sum()
    print(f"Number of repeated timestamps before adjustment: {original_count}")

    time_delta = timedelta(seconds=1 / sample_rate)
    corrected_index = []
    last_time = None
    count = 0

    for current_time in data_df.index:
        if last_time is None or current_time != last_time:
            count = 0
            corrected_index.append(current_time)
        else:
            count += 1
            corrected_time = current_time + count * time_delta
            corrected_index.append(corrected_time)
        last_time = current_time

    data_df.index = corrected_index
    data_df.sort_index(inplace=True)
    adjusted_count = pd.Index(corrected_index).duplicated().sum()
    print(f"Number of repeated timestamps after adjustment: {adjusted_count}")

    return data_df

def remove_duplicates(data_df):
    """
    Remove duplicate rows based on timestamp index.
    """
    initial_duplicates = data_df.index.duplicated().sum()
    if initial_duplicates > 0:
        print(f"Found {initial_duplicates} duplicated timestamps. Removing duplicates...")
        data_df = data_df[~data_df.index.duplicated(keep='first')]
        remaining_duplicates = data_df.index.duplicated().sum()
        print(f"Remaining duplicates after removal: {remaining_duplicates}")
    else:
        print("No duplicates found.")
    return data_df

def handle_missing_data(data_df, sample_rate):
    """
    Identify long gaps in data and fill them with zeros, interpolate short gaps.
    """
    data_df['time_diff'] = data_df.index.to_series().diff().dt.total_seconds()
    long_gaps = data_df['time_diff'] > 20
    gap_starts = long_gaps.shift(1).fillna(False) & ~long_gaps

    for start in data_df.index[gap_starts]:
        end = start + pd.Timedelta(seconds=int(data_df.loc[start, 'time_diff']))
        data_df.loc[start:end, data_df.columns.difference(['time_diff'])] = 0

    data_df.interpolate(method='time', limit=20, limit_direction='both', inplace=True)
    data_df.drop(columns='time_diff', inplace=True)
    return data_df

def modify_modality_names(data, sensor_name):
    """
    Rename data columns based on sensor type to standardize naming conventions across different modalities.
    """
    if 'corsano_wrist_acc' in sensor_name:
        data = data.rename(columns={'accX': 'wrist_acc_x', 'accY': 'wrist_acc_y', 'accZ': 'wrist_acc_z'})
        return 'corsano_wrist', data
    elif 'cosinuss_ear_acc_x_acc_y_acc_z' in sensor_name:
        data = data.rename(columns={'acc_x': 'ear_acc_x', 'acc_y': 'ear_acc_y', 'acc_z': 'ear_acc_z'})
        return 'cosinuss_ear', data
    elif 'mbient_imu_wc_accelerometer' in sensor_name:
        data = data.rename(columns={'x_axis_g': 'imu_acc_x', 'y_axis_g': 'imu_acc_y', 'z_axis_g': 'imu_acc_z'})
        return 'mbient_acc', data
    elif 'mbient_imu_wc_gyroscope' in sensor_name:
        data = data.rename(columns={'x_axis_dps': 'gyro_x', 'y_axis_dps': 'gyro_y', 'z_axis_dps': 'gyro_z'})
        return 'mbient_gyro', data
    elif 'vivalnk_vv330_acceleration' in sensor_name:
        data = data.rename(columns={'x': 'vivalnk_acc_x', 'y': 'vivalnk_acc_y', 'z': 'vivalnk_acc_z'})
        return 'vivalnk_acc', data
    elif 'sensomative_bottom_logger' in sensor_name:
        data = data.rename(columns={'value_0': 'bottom_value_0', 'value_1': 'bottom_value_1', 'value_2': 'bottom_value_2', 
                                    'value_3': 'bottom_value_3', 'value_4': 'bottom_value_4', 'value_5': 'bottom_value_5', 
                                    'value_6': 'bottom_value_6', 'value_7': 'bottom_value_7', 'value_8': 'bottom_value_8',
                                    'value_9': 'bottom_value_9', 'value_10': 'bottom_value_10', 'value_11': 'bottom_value_11'})
        return 'sensomative_bottom', data
    elif 'sensomative_back_logger' in sensor_name:
        data = data.rename(columns={'value_0': 'back_value_0', 'value_1': 'back_value_1', 'value_2': 'back_value_2', 
                                    'value_3': 'back_value_3', 'value_4': 'back_value_4', 'value_5': 'back_value_5', 
                                    'value_6': 'back_value_6', 'value_7': 'back_value_7', 'value_8': 'back_value_8',
                                    'value_9': 'back_value_9', 'value_10': 'back_value_10', 'value_11': 'back_value_11'})
        return 'sensomative_back', data
    elif 'corsano_bioz_acc' in sensor_name:
        data = data.rename(columns={'accX': 'bioz_acc_x', 'accY': 'bioz_acc_y', 'accZ': 'bioz_acc_z'})
        return 'corsano_bioz', data
    else:
        return sensor_name, data

def get_sensor_columns(sensor_name):
    """
    Return a list of data columns for a given sensor type, based on previous naming adjustments.
    """
    sensor_columns = {
        'corsano_wrist_acc': ['wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z'],
        'cosinuss_ear_acc_x_acc_y_acc_z': ['ear_acc_x', 'ear_acc_y', 'ear_acc_z'],
        'mbient_imu_wc_accelerometer': ['imu_acc_x', 'imu_acc_y', 'imu_acc_z'],
        'mbient_imu_wc_gyroscope': ['gyro_x', 'gyro_y', 'gyro_z'],
        'vivalnk_vv330_acceleration': ['vivalnk_acc_x', 'vivalnk_acc_y', 'vivalnk_acc_z'],
        'sensomative_bottom_logger': ['bottom_value_0', 'bottom_value_1', 'bottom_value_2', 'bottom_value_3', 'bottom_value_4', 'bottom_value_5', 'bottom_value_6', 'bottom_value_7', 'bottom_value_8', 'bottom_value_9', 'bottom_value_10', 'bottom_value_11'],
        'sensomative_back_logger': ['back_value_0', 'back_value_1', 'back_value_2', 'back_value_3', 'back_value_4', 'back_value_5', 'back_value_6', 'back_value_7', 'back_value_8', 'back_value_9', 'back_value_10', 'back_value_11'],
        'corsano_bioz_acc': ['bioz_acc_x', 'bioz_acc_y', 'bioz_acc_z']
    }
    return sensor_columns.get(sensor_name, [])

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def resample_and_filter(data, upsample_freq, downsample_freq, lowcut, highcut, filter_order):
    """
    Resample and filter data. Remove rows with very small time differences, interpolate, and apply a lowpass filter.
    """
    # Detect small time differences and filter them out
    time_diffs = np.diff(data.index.astype('int64'))
    mask = time_diffs < 100000  # Mask for differences less than 100,000 nanoseconds
    indices_to_keep = ~np.hstack([False, mask])  # Maintain array length consistency as diff is shorter by one element
    cleaned_data = data.iloc[indices_to_keep]
    # Generate upsample time series
    time_upsample = pd.date_range(start=cleaned_data.index[0], end=cleaned_data.index[-1], freq=pd.Timedelta(seconds=1 / upsample_freq))

    # Apply linear interpolation
    resampled_data = {}
    for column in cleaned_data.columns:
        f = interp1d(cleaned_data.index.astype('int64'), cleaned_data[column].values, kind='slinear', fill_value="extrapolate")
        resampled_data[column] = f(time_upsample.astype('int64'))

    # Convert dictionary to DataFrame
    resampled_df = pd.DataFrame(resampled_data, index=time_upsample)

    # Apply a lowpass filter
    b, a = butter_lowpass(10, upsample_freq, filter_order)
    filtered_data = {col: filtfilt(b, a, resampled_df[col].values) for col in resampled_df.columns}
    filtered_df = pd.DataFrame(filtered_data, index=time_upsample)

    # Process very small values, setting a threshold, values smaller than 1e-6 are considered as zero
    filtered_df[filtered_df.abs() < 1e-6] = 0

    # Downsampling
    downsample_interval = pd.Timedelta(seconds=1 / downsample_freq)
    downsampled_data = filtered_df.resample(downsample_interval).mean()

    return downsampled_data
def ensure_unique_timestamps(data_df, context="Checking data"):
    """
    Ensure that there are no duplicate timestamps in the data DataFrame.
    """
    duplicates_count = data_df.index.duplicated().sum()
    if duplicates_count > 0:
        print(f"{context}: Found {duplicates_count} duplicated timestamps. Removing duplicates...")
        data_df = data_df[~data_df.index.duplicated(keep='first')]
    else:
        print(f"{context}: No duplicates found.")
    return data_df

def filter_and_fill_data(data_df, start_time, end_time, downsample_freq):
    """
    Filter data within the specified start and end time and fill missing indices.
    """
    filtered_data = data_df[start_time:end_time]
    
    full_index = pd.date_range(start=start_time, end=end_time, freq=pd.Timedelta(seconds=1/upsample_freq))
    filtered_data = filtered_data.reindex(full_index, fill_value=0)
    
    return filtered_data

def select_data_loader(sensor_name):    
    """
    Select the appropriate data loader based on the sensor name.
    """
    if sensor_name == 'vivalnk_vv330_acceleration':
        return data_loader_with_dir
    elif sensor_name in ['mbient_imu_wc_accelerometer', 'mbient_imu_wc_gyroscope']:
        return data_loader_no_dir_ms
    else:
        return data_loader_no_dir

def concatenate_sensor_data(subject_dir, raw_data_pars, start_time, end_time, downsample_freq, subject):
    """
    Concatenate sensor data for all sensors for a given subject, process it, and handle missing and duplicated data.
    """
    all_sensors_data = {}
    full_time_index = pd.date_range(start=start_time, end=end_time, freq=pd.Timedelta(seconds=1/upsample_freq))

    for sensor_name, sensor_params in raw_data_pars.items():
        print(f"Processing {sensor_name}...")
        sys.stdout.flush()
        data_loader = select_data_loader(sensor_name)
        sensor_data, time_ranges = data_loader(subject_dir, sensor_name, sensor_params)

        if sensor_data.empty:
            print(f"No data available for sensor {sensor_name}, adding zero-filled data...")
            zero_data = pd.DataFrame(0, index=full_time_index, columns=get_sensor_columns(sensor_name))
            all_sensors_data[sensor_name] = zero_data
            print(all_sensors_data[sensor_name])
            continue
        if subject == 'OutSense-515':
            sensor_data.index += timedelta(days=1)
            
        processed_data = process_modality(sensor_data, sensor_params['sample_rate'])
        processed_data = remove_duplicates(processed_data)
        processed_data = handle_missing_data(processed_data, sensor_params['sample_rate'])
        
        # Modify modality names after ensuring there are no NaN values
        new_name, modified_data = modify_modality_names(processed_data, sensor_name)
        
        # Up-sampling, Filtering, and Down-sampling the Data
        final_processed_data = resample_and_filter(modified_data, upsample_freq, downsample_freq, lowcut_kinematic, highcut_kinematic, filter_order)
        
        # Filter data based on manual log timestamps and fill missing data 
        final_processed_data_filled = filter_and_fill_data(final_processed_data, start_time, end_time, downsample_freq)
        
        # Store the final processed data
        all_sensors_data[new_name] = final_processed_data_filled

    combined_data = pd.concat(all_sensors_data.values(), axis=1)
    combined_data.index.name = 'time'
    return combined_data

def get_activity_filename(subject):
    """
    Determine the filename for activity data based on the subject's unique identifier.
    """
    if '425_48h' in subject:
        return 'final_activities_425.xlsx'
    else:
        subject_num = ''.join(filter(str.isdigit, subject))
        return f'final_activities_{subject_num}.xlsx'
    
def process_subject(subject, current_dir, raw_data_pars, subjects):
    """
    Process data for a single subject, including loading manual log data and activity labels.
    """
    subject_dir = os.path.join(current_dir, subject)
    manual_log_dir = os.path.join(subject_dir, "manual_log")
    manual_log_file = os.path.join(manual_log_dir, "manual_log.csv.gz")
    
    if not os.path.exists(manual_log_file):
        print(f"Manual log file does not exist for {subject}.")
        return None

    label_file_path = os.path.join('activity_label_10s', get_activity_filename(subject))
    if not os.path.exists(label_file_path):
        print(f"Activity label file does not exist for {subject}.")
        return None
    
    activity_labels = pd.read_excel(label_file_path)
    activity_labels['Start Time'] = pd.to_datetime(activity_labels['Start Time'], format='%Y-%m-%d %H:%M:%S')
    activity_labels['End Time'] = pd.to_datetime(activity_labels['End Time'], format='%Y-%m-%d %H:%M:%S')

    min_start_time = activity_labels['Start Time'].min()
    max_end_time = activity_labels['End Time'].max()
    print(f'Start Time: {min_start_time}, End Time: {max_end_time}')
    sys.stdout.flush()
    
    if subject == 'OutSense-515':
        min_start_time += timedelta(days=1)
        max_end_time += timedelta(days=1)
        activity_labels['Start Time'] += timedelta(days=1)
        activity_labels['End Time'] += timedelta(days=1)
        print(f"Time adjusted by +1 day for subject {subject}. Start Time: {min_start_time}, End Time: {max_end_time}")

    final_data = concatenate_sensor_data(subject_dir, raw_data_pars, min_start_time, max_end_time, params['downsample_freq'], subject)

    final_data['Activity'] = 'not recognized'  # default label
    for _, row in activity_labels.iterrows():
        mask = (final_data.index >= row['Start Time']) & (final_data.index <= row['End Time'])
        final_data.loc[mask, 'Activity'] = row['Activity']

    # Collect unique activities for the subject
    unique_activities = activity_labels['Activity'].unique()
    print(f"Activities for {subject}: {unique_activities}")

    # Assign a numerical index for each subject based on their order in the list
    final_data['Subject'] = subjects.index(subject)  # Ensure 'subjects' is accessible
    print(subjects.index(subject))
    
    # Get the required sensor columns from settings and add additional columns for activity and subject
    sensor_columns = [col for sensor in raw_data_pars for col in get_sensor_columns(sensor)]
    required_columns = sensor_columns + ['Activity', 'Subject']
    final_data = final_data[required_columns]

    return final_data


"""
Main Execution
"""
def main():
    subjects = [
        'OutSense-036', 'OutSense-284', 'OutSense-425_48h', 'OutSense-498', 'OutSense-515',
        'OutSense-532', 'OutSense-619', 'OutSense-694', 'OutSense-785'
    ]
    combined_data = pd.DataFrame()

    # Process each subject and combine their data into a single DataFrame
    for subject in subjects:
        subject_index = subjects.index(subject)
        
        print(f"Processing data for {subject}...")
        print(f"Subject {subject} is assigned index number {subject_index}.")
        
        subject_data = process_subject(subject, current_dir, params['raw_data_pars'], subjects)
        print(subject_data)
        if subject_data is not None:
            subject_data['Subject'] = subject_index
            combined_data = pd.concat([combined_data, subject_data])

    # Check if any data was processed and combined
    if not combined_data.empty:
        print("Activities before renaming:")
        print(combined_data['Activity'].unique())
        print(f"Total distinct activities before renaming: {len(combined_data['Activity'].unique())}")

        renaming_map = {
            'sit_to_wheelchair': 'sit_bed_to_wheelchair',
            'wheelchair_to_bed': 'wheelchair_to_sit_bed',
            'reading_newspaper': 'reading',
            'using_phones': 'using_phone',
            'wheelchair_to_sitting_bed': 'wheelchair_to_sit_bed',
            'sitting_bed_to_wheelchair': 'sit_bed_to_wheelchair',
            'turning_bed_counter_clockwise': 'not recognized',
            'toilet_rountine': 'toilet_routine',
            'clearning_teeth': 'not recognized',
            'driving': 'not recognized',
            'assisted_propulsion_and_phone': 'assisted_propulsion_and_using_phone'
        }
        combined_data['Activity'] = combined_data['Activity'].replace(renaming_map)
        
        print("Activities after renaming:")
        print(combined_data['Activity'].unique())
        print(f"Total distinct activities after renaming: {len(combined_data['Activity'].unique())}")

        activity_to_number = {activity: idx for idx, activity in enumerate(combined_data['Activity'].unique())}
        print("Activity to number mapping:")
        for activity, number in activity_to_number.items():
            print(f"{activity}: {number}")

        combined_data['Activity'] = combined_data['Activity'].map(activity_to_number)
        
        for subject in subjects:
            subject_activities = combined_data[combined_data['Subject'] == subjects.index(subject)]['Activity'].unique()
            print(f"Subject {subject} has activity labels: {subject_activities}, {len(subject_activities)}")
        
        print("\nNumber of subjects per activity label:")
        for activity, number in activity_to_number.items():
            subjects_with_activity = combined_data[combined_data['Activity'] == number]['Subject'].unique()
            print(f"Activity '{activity}' has {len(subjects_with_activity)} subjects: {subjects_with_activity}")

        combined_data['ChangePoint'] = combined_data['Activity'].shift(1) != combined_data['Activity']
        combined_data.loc[0, 'ChangePoint'] = True

        activity_counts = combined_data['Activity'].value_counts()
        print("Previous Activity counts:")
        print(activity_counts)

        combined_data = combined_data[:-1]
        if not isinstance(combined_data.index, pd.DatetimeIndex):
            print('confirm datatimeindex')
            combined_data.index = pd.to_datetime(combined_data.index)
        combined_data.loc[combined_data.index[-1], 'ChangePoint'] = True

        true_count = combined_data['ChangePoint'].sum()
        print(f"Previous Total number of change points: {true_count}")
        combined_data = combined_data[combined_data['Activity'] != 1]

        # Update the activity map, remove the activity numbered 1, and reduce all tags greater than 1 by 1
        activity_mapping = {
            0: "sit_bed_to_wheelchair",
            1: "conversation",
            2: "self_propulsion_and_conversation",
            3: "self_propulsion",
            4: "sitting_wheelchair",
            5: "resting",
            6: "arm_raises",
            7: "chair_to_wheelchair",
            8: "using_computer",
            9: "using_phone",
            10: "assisted_propulsion",
            11: "washing_hands",
            12: "changing_clothes",
            13: "pressure_relief",
            14: "eating",
            15: "put_on_clothes",
            16: "toilet_routine",
            17: "wheelchair_to_car",
            18: "cycling",
            19: "manipulating",
            20: "take_off_clothes",
            21: "reading",
            22: "drinking",
            23: "lying",
            24: "transfer",
            25: "bending",
            26: "housework",
            27: "folding_clothes",
            28: "skin_care",
            29: "beard_hair_styling",
            30: "preparing_meal",
            31: "watching_tv",
            32: "brushing_teeth",
            33: "lying_to_sit",
            34: "cleaning_teeth",
            35: "washing_face",
            36: "conversation_and_phones",
            37: "assisted_propulsion_and_using_phone",
            38: "sitting_chair",
            39: "conversation_and_eatting",
            40: "sit_to_lying",
            41: "car_to_wheelchair",
            42: "writing",
            43: "wheelchair_to_sit_bed",
            44: "putting_toothpaste",
            45: "conversation_and_drinking",
            46: "rinsing_mouth",
            47: "using_phone_and_eatting",
            48: "shampoo"
        }

        # Remapping
        combined_data['Activity'] = combined_data['Activity'].apply(lambda x: x-1 if x > 1 else x)

        new_mapping = {k: activity_mapping[k] for k in sorted(activity_mapping.keys()) if k in combined_data['Activity'].unique()}
        print("New activity mapping:")
        for k, v in new_mapping.items():
            print(f"{k}: {v}")

        combined_data.loc[combined_data.index[0], 'ChangePoint'] = False
        combined_data.loc[combined_data.index[-1], 'ChangePoint'] = False

        combined_data['Activity'] = combined_data['Activity'].astype(int)
        combined_data['Subject'] = combined_data['Subject'].astype(int)

        activity_counts = combined_data['Activity'].value_counts()
        print("After Activity counts:")
        print(activity_counts)

        true_count = combined_data['ChangePoint'].sum()
        print(f"After Total number of change points: {true_count}")

        combined_data.to_pickle('100_sensor_data_interp1d_lowpass_activity_boundary_wo_not_recognized.pkl')
        print('已经保存上采样到100hz的data, wo not recognized')

    else:
        print("No data was processed.")

if __name__ == '__main__':
    start_time = time.time() 
    main()
    end_time = time.time() 
    total_time_seconds = end_time - start_time
    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60) 
    print(f"\nTotal Execution Time: {hours} hours and {minutes} minutes")