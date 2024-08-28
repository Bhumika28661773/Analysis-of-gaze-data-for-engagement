# Analysis-of-gaze-data-for-engagement

# Packages needed to run the code

pip install opencv-python
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
pip install -e ".[demo]"
cd checkpoints && \
./download_ckpts.sh && \
cd ..
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# How to run the file ?

To run the experiment:-
1. convert hdfy file to csv file by inputing the hdfy file name in "hdfyCsvConvert.py" and running it. The file extracted will contain the gaze data and other information.
2. next run the "overlayGaze.py" to overlay gaze points on the video.
3. Then to convert the video into videoframes run "convertToVideoFrames.py" file.
4. Then run "mainCode.py", to start segmentation and tracking. After running prompts will be asked. If "e" is inputted, then program will exit. First, one frame would be displayed to identify object to mark ID 1, after adding the coordinate, prompt will be displayed again. That time, if "y" is inputted, multiple coordinates can be added to same or different frame by passing frame number and ID. This step can be done multiple times. Once after selecting all the segments, if "n" is inputted, the segments will be progressed across the video frames to segment objects and to identify where gaze falls.
5. To get a csv file containing the analysis results, run "analyseGazeSegments.py" file.
6. Run "User_Engagement_Analysis.ipnb" file to analyse the gaze data


