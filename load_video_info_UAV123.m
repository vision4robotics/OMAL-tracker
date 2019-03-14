function seq = load_video_info_UAV123(video_base_path, video, ground_truth_base_path)

    seqs=configSeqs;
    seq.video_path = strcat(video_base_path, video);
    i=1;
    while ~strcmpi(seqs{i}.name,video)
            i=i+1;
    end
    
    seq.VidName = seqs{i}.name;
    seq.st_frame = seqs{i}.startFrame;
    seq.en_frame = seqs{i}.endFrame;
    
    seq.ground_truth_fileName = seqs{i}.name;
    ground_truth = dlmread([ground_truth_base_path seq.ground_truth_fileName '.txt']);
    
    seq.ground_truth = ground_truth;
    seq.len = seq.en_frame-seq.st_frame+1;
    seq.init_rect = ground_truth(1,:);
    
    img_path = [video_base_path video];
    img_files = dir(fullfile(img_path, '*.jpg'));
    img_files = {img_files.name};
    seq.s_frames_temp = cellstr(img_files);
    seq.s_frames = seq.s_frames_temp(1, seq.st_frame : seq.en_frame);