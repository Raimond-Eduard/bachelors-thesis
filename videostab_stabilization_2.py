import cv2 as cv
from vidstab import VidStab

# Input and output file names
input_path = 'Inputs/Podu_Ros.mp4'
output_path = 'Outputs/stabilization_with_videostab_4.avi'

# Parameters for better stabilization
smoothing_window = 200
show_preview = True
keypoint_method = 'GFTT'
border_type = 'black'

stabilizer = VidStab(kp_method=keypoint_method)

stabilizer.stabilize(
    input_path=input_path,
    output_path=output_path,
    smoothing_window=smoothing_window,
    border_type=border_type,
    show_progress=True
)

