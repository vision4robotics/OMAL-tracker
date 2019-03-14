clear; 
clc;
close all;
%-------------UAV123-----------------
video_base_path = '.\UAV123_10fps\data_seq\UAV123_10fps\';
ground_truth_base_path = '.\UAV123_10fps\anno\UAV123_10fps\';
video = choose_video(video_base_path);
seq = load_video_info_UAV123(video_base_path, video, ground_truth_base_path);
video_path = seq.video_path;
ground_truth = seq.ground_truth;

% Run OMAL-main function
learning_rate = 0.013;  % you can use different learning rate.
results       = run_OMAL(seq, video_path, learning_rate, 0.013, 0.005);
  
% save results
result_name = video;
OMAL = results;

% save results to specified folder
savedir = '.\result\';
save([savedir,result_name],'OMAL');

% plot precision figure
precision_plot(results.res,ground_truth, video, savedir,1);
