import os
import time
import ffmpeg
import pandas as pd

# Path to the video files folder
folder_path = '/scai_data2/scai_datasets/raw/scai-outsense/OutSense-036/gopro_camera/SD2/'

# Path to the Excel file
excel_file_path = './Data Labeling/OutSense-036.xlsx'

# Retrieve all MP4 files from the folder and sort them by filename
video_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.MP4')])
print(f"Total MP4 files found: {len(video_files)}")

# Load the Excel file
xls = pd.ExcelFile(excel_file_path)

# Initialize a counter for successfully processed video files
successful_count = 0

# Process each video file
for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    
    try:
        # Use ffmpeg.probe to retrieve video file information
        vid = ffmpeg.probe(video_path)

        # Extract information about the video stream
        video_stream = None
        for stream in vid['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break

        if video_stream:
            # Extract required information from the video stream
            frame_rate_str = video_stream['r_frame_rate']
            duration = float(video_stream['duration'])
            frame_count = int(video_stream['nb_frames'])
            creation_time = video_stream['tags'].get('creation_time', 'N/A')
            encoded_date = video_stream['tags'].get('encoded_date', 'N/A')
            
            # Calculate the frame rate
            num, den = map(int, frame_rate_str.split('/'))
            frame_rate = num / den

            # Retrieve creation and modification time from the file system
            file_creation_time = os.path.getctime(video_path)
            file_modification_time = os.path.getmtime(video_path)

            # Calculate video start time
            start_time = file_modification_time - duration

            # Format time for readability
            file_creation_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_creation_time))
            file_modification_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_modification_time))
            start_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

            # Video file base name without extension
            video_base_name = video_file.split('.')[0]

            # Find matching Excel sheets
            matched_sheets = [s for s in xls.sheet_names if s.split('_')[-1] == video_base_name]

            if matched_sheets:
                # Assuming there's exactly one matching sheet
                sheet_name = matched_sheets[0]
                
                # Read the matching sheet
                df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

                # Remove the first column
                df.drop(df.columns[0], axis=1, inplace=True)

                # Add two new columns with start and end times
                new_cols = pd.DataFrame({
                    '': ['start time', 'end time'],
                    ' ': [start_time_formatted, file_modification_time_formatted]
                })

                # Combine new DataFrame with existing data
                df = pd.concat([new_cols, df], axis=1)

                # Write back to Excel
                with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                print(f"Processed sheet: {sheet_name}")

            # Increment the successful file processing counter
            successful_count += 1
        else:
            print(f"No video stream found in the file {video_file}.")
    except ffmpeg.Error as e:
        print(f"An error occurred while processing file {video_file}: {e.stderr.decode()}")
    except Exception as e:
        print(f"An error occurred while processing Excel: {str(e)}")

# Print the count of successfully processed video files
print(f"Successfully processed {successful_count} out of {len(video_files)} MP4 files.")
