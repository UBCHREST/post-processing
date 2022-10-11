clear all; close all; clc;
vid = VideoReader('run1.mov');%name of the video
numFrames = vid.NumberOfFrames;
n = numFrames;
for i 0:500 %set the frames 
    frames = read(vid,i);

    imwrite(frames,['Img' int2str(i), '.tiff']);
end