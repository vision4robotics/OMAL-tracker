    % This function implements the DCP_BACF tracker.
function [results] = OMAL_optimized(params, half_learningrate, half_worse_learningrate)
%% 
%% Setting parameters for part-based trackers.
search_area_scale   = 5; 
output_sigma_factor = params.output_sigma_factor; 
learning_rate       = params.learning_rate; 
filter_max_area     = params.filter_max_area; 
nScales             = params.number_of_scales; 
scale_step          = params.scale_step;
interpolate_response = params.interpolate_response; 
n_sample = params.numsample;
affsig = params.affsig;
features       = params.t_features;
video_path     = params.video_path; 
s_frames       = params.s_frames; 
pos            = floor(params.init_pos); 
target_sz      = floor(params.wsize);
visualization       = params.visualization; 
num_frames          = params.no_fram; 
init_target_sz      = target_sz; 
sub_target_sz = floor(target_sz/2);
% The number of subregions
subw_num = 4; 
sub_init_pos = zeros(subw_num,2);
gamma = 1.0e-1;
% Set the feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size; 
search_area = prod(init_target_sz/ featureRatio * search_area_scale); 
% When the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh') 
    if search_area < params.t_global.cell_selection_thresh * filter_max_area 
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end
end
global_feat_params = params.t_global;

if search_area > filter_max_area 
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end
% Target size at the initial scale
base_target_sz = target_sz / currentScaleFactor; 
% Window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);                    % proportional area, same aspect ratio as the target
    case 'square' 
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2);  % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end
% Set the size to exactly match the cell size
sz = round(sz / (featureRatio*2)) *featureRatio; 
use_sz = floor(sz/featureRatio);
% Construct the label function- correlation output, 2D gaussian function,
% With a peak located upon the target
output_sigma = sqrt(prod(floor(base_target_sz/2/(featureRatio)))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid( rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf           = fft2(y); %   FFT of y.
if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else
    interp_sz = use_sz;
end
% Construct cosine window 
cos_window = single(hann(floor(use_sz(1)))*hann(floor(use_sz(2)))');
% Calculate feature dimension 
try 
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try 
        im = imread(s_frames{1});
    catch
        %disp([video_path '/' s_frames{1}])
        im = imread([video_path '/' s_frames{1}]);
    end
    
end

if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end

% Compute feature dimensionality 
feature_dim = 0;
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor') 
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray') 
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
        feature_dim = feature_dim + features{n}.fparams.nDim;
    end
end

if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end

if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)); 
    scaleFactors = scale_step .^ scale_exp; 
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end
if interpolate_response >= 3
    % Pre-computes the grid that is used for socre optimization
    ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
    kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end
% Allocate memory for multi-scale tracking 
multires_pixel_template = zeros(sz(1), sz(2), size(im,3), nScales, 'uint8'); 
small_filter_sz = floor(base_target_sz/featureRatio); 
%% 
%% smaller image filter parameter configuration
search_area_scale_half   = 5;
filter_max_area_half     = params.filter_max_area;
scale_step_half          = params.scale_step;
pos_half         = floor(params.init_pos/2);
target_sz_half   = floor(params.wsize/2);
init_target_sz_half = target_sz_half;
search_area_half = prod(init_target_sz_half / featureRatio * search_area_scale_half);
% when the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh')
    if search_area_half < params.t_global.cell_selection_thresh * filter_max_area_half
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz_half * search_area_scale_half)/(params.t_global.cell_selection_thresh * filter_max_area_half)))));
        
        featureRatio = params.t_global.cell_size;
        search_area_half = prod(init_target_sz_half / featureRatio * search_area_scale_half);
    end
end
global_feat_params_half = params.t_global;
if search_area_half > filter_max_area_half
    currentScaleFactor_half = sqrt(search_area_half / filter_max_area_half);
else
    currentScaleFactor_half = 1.0;
end
% target size at the initial scale
base_target_sz_half = target_sz_half / currentScaleFactor_half;
% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz_half = floor( base_target_sz_half * search_area_scale_half);     % proportional area, same aspect ratio as the target
    case 'square'
        sz_half = repmat(sqrt(prod(base_target_sz_half * search_area_scale_half)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz_half = base_target_sz_half + sqrt(prod(base_target_sz_half * search_area_scale_half) + (base_target_sz_half(1) - base_target_sz_half(2))/4) - sum(base_target_sz_half)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end
% set the size to exactly match the cell size
sz_half = round(sz_half / featureRatio) * featureRatio;
use_sz_half = floor(sz_half/featureRatio);
% construct the label function- correlation output, 2D gaussian function,
% with a peak located upon the target
output_sigma_half = sqrt(prod(floor(base_target_sz_half/featureRatio))) * output_sigma_factor;
rg_half           = circshift(-floor((use_sz_half(1)-1)/2):ceil((use_sz_half(1)-1)/2), [0 -floor((use_sz_half(1)-1)/2)]);
cg_half           = circshift(-floor((use_sz_half(2)-1)/2):ceil((use_sz_half(2)-1)/2), [0 -floor((use_sz_half(2)-1)/2)]);
[rs_half, cs_half]     = ndgrid( rg_half,cg_half);
y_half            = exp(-0.5 * (((rs_half.^2 + cs_half.^2) / output_sigma_half^2)));
yf_half           = fft2(y_half); %   FFT of y.
if interpolate_response == 1
    interp_sz_half = use_sz_half * featureRatio;
else
    interp_sz_half = use_sz_half;
end
% construct cosine window
cos_window_half = single(hann(use_sz_half(1))*hann(use_sz_half(2))');

if nScales > 0
    scale_exp_half = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors_half = scale_step_half .^ scale_exp_half;
    min_scale_factor_half = scale_step_half ^ ceil(log(max(5 ./ sz_half)) / log(scale_step_half));
    max_scale_factor_half = scale_step_half ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz_half)) / log(scale_step_half));
end

if interpolate_response >= 3
    % Pre-computes the grid that is used for socre optimization
    ky_half = circshift(-floor((use_sz_half(1) - 1)/2) : ceil((use_sz_half(1) - 1)/2), [1, -floor((use_sz_half(1) - 1)/2)]);
    kx_half = circshift(-floor((use_sz_half(2) - 1)/2) : ceil((use_sz_half(2) - 1)/2), [1, -floor((use_sz_half(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end

% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
time = 0;
% allocate memory for multi-scale tracking
multires_pixel_template_half = zeros(sz_half(1), sz_half(2), size(im,3), nScales, 'uint8');
small_filter_sz_half = floor(base_target_sz_half/featureRatio);

%% main process of the code
loop_frame = 1;
% theta_thresh=5;
theta_history=[];
theta_all = [];
meann = [];
th = 0;
pos_result = pos;
target_sz_result = target_sz;

for frame = 1:numel(s_frames)
    % Load image
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    % pyramid model using to coarse estimation
    im_pyramid = imresize(im,0.5);
    %im_pyramid = imresize(im, 0.5, 'nearest');
%     tepp = im(:,:,1);
%     tep = im_pyramid(:,:,1);
    tic(); 
    
    %%
    %do not estimate translation and scaling on the first frame, since we
    %just want to initialize the tracker here
    if frame == 1
       %% initializing the filter
       %% initializing the whole-filter of the small image 
        % extract training sample image region
        pixels_half = get_pixels(im_pyramid,pos_half,round(sz_half*currentScaleFactor_half),sz_half);
        % extract features and do windowing
        xf_half = fft2(bsxfun(@times,get_features(pixels_half,features,global_feat_params_half),cos_window_half));
        % surface model initialize
        model_xf_half = xf_half;
        % ADMM method to train a init filter
        g_f_half = single(zeros(size(xf_half)));
        h_f_half = g_f_half;
        l_f_half = g_f_half;
        mu_half    = 1;
        betha_half = 10;
        mumax_half = 10000;
        i = 1;
        T_half = prod(use_sz_half);
        S_xx_half = sum(conj(model_xf_half) .* model_xf_half, 3);
        params.admm_iterations = 2;
        while (i <= params.admm_iterations)
            %   solve for G- please refer to the paper for more details
            B_half = S_xx_half + (T_half * mu_half);
            S_lx_half = sum(conj(model_xf_half) .* l_f_half, 3);
            S_hx_half = sum(conj(model_xf_half) .* h_f_half, 3);
            g_f_half = (((1/(T_half*mu_half)) * bsxfun(@times, yf_half, model_xf_half)) - ((1/mu_half) * l_f_half) + h_f_half) - ...
            bsxfun(@rdivide,(((1/(T_half*mu_half)) * bsxfun(@times, model_xf_half, (S_xx_half .* yf_half))) - ((1/mu_half) * bsxfun(@times, model_xf_half, S_lx_half)) + (bsxfun(@times, model_xf_half, S_hx_half))), B_half);
            %   solve for H
            h_half = (T_half/((mu_half*T_half)+ params.admm_lambda))* ifft2((mu_half*g_f_half) + l_f_half);
            [sx_half,sy_half,h_half] = get_subwindow_no_window(h_half, floor(use_sz_half/2) , small_filter_sz_half);
            t_half = single(zeros(use_sz_half(1), use_sz_half(2), size(h_half,3)));
            t_half(sx_half,sy_half,:) = h_half;
            h_f_half = fft2(t_half);
            %   update L
            l_f_half = l_f_half + (mu_half * (g_f_half - h_f_half));
            %   update mu- betha = 10.
            mu_half = min(betha_half * mu_half, mumax_half);
            i = i+1;
        end
        % initializing the four part-based filters
        %target_sz = base_target_sz * currentScaleFactor;
         sub_target_sz = target_sz/2;
         % Location of sub-regions
         sub_init_pos(1,:) = pos - floor(sub_target_sz/2);
         sub_init_pos(2,:) = pos + [-floor(sub_target_sz(1)/2), floor(sub_target_sz(2)/2)];
         sub_init_pos(3,:) = pos + [floor(sub_target_sz(1)/2), -floor(sub_target_sz(2)/2)];
         sub_init_pos(4,:) = pos + floor(sub_target_sz/2);
         old_sub_pos = sub_init_pos;
         for num = 1:subw_num
             % Extract training sample image region 
             pixels = get_pixels(im,sub_init_pos(num,:),round(sz*currentScaleFactor),sz);
             % Extract features and do windowing 
             xf = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params),cos_window));
             model_xf{num} = xf;
        
             % ADMM intialization
             g_f = single(zeros(size(xf))); 
             h_f = g_f;
             l_f = g_f;
             mu    = 1;
             betha = 10;
             mumax = 10000;
             i = 1;
             T = prod(use_sz/2); 
             S_xx = sum(conj(model_xf{num}) .* model_xf{num}, 3); 
             params.admm_iterations = 2; 
             % ADMM iteration
             while (i <= params.admm_iterations)
                 % Solve for G- please refer to the paper for more details
                 B = S_xx + (T * mu);
                 S_lx = sum(conj(model_xf{num}) .* l_f, 3); %xl
                 S_hx = sum(conj(model_xf{num}) .* h_f, 3); %xh
                 g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf{num})) - ((1/mu) * l_f) + h_f) - ...
                     bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf{num}, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf{num}, S_lx)) + (bsxfun(@times, model_xf{num}, S_hx))), B);

                 % Solve for H
                 h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f); 
                 [sx,sy,h] = get_subwindow_no_window(h, floor((use_sz)/2) , small_filter_sz);
                 t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
                 t(sx,sy,:) = h; 
                 h_f = fft2(t);  
                 % Update L 
                l_f = l_f + (mu * (g_f - h_f));
                % Update mu- betha = 10.
                mu = min(betha * mu, mumax); % 
                i = i+1;
             end 
             % Save the g_f of sub-regions
             sub_g_f(:,:,:,num) = g_f;  
         end   
    %% if frame > 2
    else
       %% small image detection
        %course estimation
        %using the half image
        for scale_ind = 1:nScales
            multires_pixel_template_half(:,:,:,scale_ind) = ...
                get_pixels(im_pyramid, pos_half, round(sz_half*currentScaleFactor_half*scaleFactors_half(scale_ind)), sz_half);
        end
        xtf_half = fft2(bsxfun(@times,get_features(multires_pixel_template_half,features,global_feat_params_half),cos_window_half));
        responsef_half = permute(sum(bsxfun(@times, conj(g_f_half), xtf_half), 3), [1 2 4 3]);
        % if we undersampled features, we want to interpolate the
        % response so it has the same size as the image patch
        if interpolate_response == 2
            % use dynamic interp size
            interp_sz_half = floor(size(y_half) * featureRatio * currentScaleFactor_half);
        end
        responsef_padded_half = resizeDFT2(responsef_half, interp_sz_half);
        % response in the spatial domain
        response_half = ifft2(responsef_padded_half, 'symmetric');

        %get APCE
        [apce_half,peak_loc_half] = get_PMLR(response_half, use_sz_half, nScales);
        theta = mean(apce_half);

        %weighting every response map
        for n = 1:nScales
            modify_response_half(:,:,n) = bsxfun(@times, fftshift(response_half(:,:,n)), cos_mask(cos_window_half, peak_loc_half(n,:)));%
        end 
        rough_response(:,:,:) = interp_res(modify_response_half, nScales, featureRatio);
        [~,sc_inx]= max(sum(sum(rough_response,1),2));
        half_map = rough_response(:,:,sc_inx);
        
        % find maximum peak
        if interpolate_response == 3
            error('Invalid parameter value for interpolate_response');
        elseif interpolate_response == 4
            [disp_row, disp_col, sind] = resp_newton_sca(response_half, responsef_padded_half, newton_iterations, ky_half, kx_half, use_sz_half);
        else
            [row, col, sind] = ind2sub(size(response_half), find(response_half == max(response_half(:)), 1));
            disp_row = mod(row - 1 + floor((interp_sz_half(1)-1)/2), interp_sz_half(1)) - floor((interp_sz_half(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz_half(2)-1)/2), interp_sz_half(2)) - floor((interp_sz_half(2)-1)/2);
        end
        % calculate translation
        switch interpolate_response
            case 0
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor_half * scaleFactors_half(sind));
            case 1
                translation_vec = round([disp_row, disp_col] * currentScaleFactor_half * scaleFactors_half(sind));
            case 2
                translation_vec = round([disp_row, disp_col] * scaleFactors_half(sind));
            case 3
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor_half * scaleFactors_half(sind));
            case 4
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor_half * scaleFactors_half(sind));
        end
        % set the scale
        currentScaleFactor_half = currentScaleFactor_half * scaleFactors_half(sind);
        % adjust to make sure we are not to large or to small
        if currentScaleFactor_half < min_scale_factor
            currentScaleFactor_half = min_scale_factor;
        elseif currentScaleFactor_half > max_scale_factor
            currentScaleFactor_half = max_scale_factor;
        end
        
        theta = mean(apce_half);
        theta_history = [theta_history theta];
        theta_all = [theta_all theta];
        if frame > 20
            theta_history=[theta_history((end-19):end) theta];   
        end
        theta_mean = mean(theta_history);
        meann = [meann theta_mean];

        theta_thresh = 0.8*theta_mean;

        target_sz_result =  2*base_target_sz_half*currentScaleFactor_half;
       %%  ensure the confidence of small filter update position and output to the part-based model
        if theta > theta_thresh
            old_pos_half = pos_half;
            pos_half = pos_half + translation_vec;
            
%             target_sz = base_target_sz*currentScaleFactor;
            % output the result
            pos_result = 2*pos_half;
            %target_sz_result = target_sz;    
            
            pos = pos_result;
             pixels_half = get_pixels(im_pyramid,pos_half,round(sz_half*currentScaleFactor_half),sz_half);
             % extract features and do windowing
             xf_half = fft2(bsxfun(@times,get_features(pixels_half,features,global_feat_params_half),cos_window_half));
             model_xf_half = ((1 - half_learningrate) * model_xf_half) + (half_learningrate * xf_half);
             g_f_half = single(zeros(size(xf_half)));
             h_f_half = g_f_half;
             l_f_half = g_f_half;
             mu_half    = 1;
             betha_half = 10;
             mumax_half = 10000;
             i = 1;
             T_half = prod(use_sz_half);
             S_xx_half = sum(conj(model_xf_half) .* model_xf_half, 3);
             params.admm_iterations = 2;
             %ADMM
             while (i <= params.admm_iterations)
                %   solve for G- please refer to the paper for more details
                B_half = S_xx_half + (T_half * mu_half);
                S_lx_half = sum(conj(model_xf_half) .* l_f_half, 3);
                S_hx_half = sum(conj(model_xf_half) .* h_f_half, 3);
                g_f_half = (((1/(T_half*mu_half)) * bsxfun(@times, yf_half, model_xf_half)) - ((1/mu_half) * l_f_half) + h_f_half) - ...
                    bsxfun(@rdivide,(((1/(T_half*mu_half)) * bsxfun(@times, model_xf_half, (S_xx_half .* yf_half))) - ((1/mu_half) * bsxfun(@times, model_xf_half, S_lx_half)) + (bsxfun(@times, model_xf_half, S_hx_half))), B_half);
                %   solve for H
                h_half = (T_half/((mu_half*T_half)+ params.admm_lambda))* ifft2((mu_half*g_f_half) + l_f_half);
                [sx_half,sy_half,h_half] = get_subwindow_no_window(h_half, floor(use_sz_half/2) , small_filter_sz_half);
                t_half = single(zeros(use_sz_half(1), use_sz_half(2), size(h_half,3)));
                t_half(sx_half,sy_half,:) = h_half;
                h_f_half = fft2(t_half);
                %   update L
                l_f_half = l_f_half + (mu_half * (g_f_half - h_f_half));
                %   update mu- betha = 10.
                mu_half = min(betha_half * mu_half, mumax_half);
                i = i+1;
             end 
            
            % updating the four part-based filters
%             target_sz = target_sz_result;
            sub_target_sz = target_sz/2;
            % Location of sub-regions
            sub_init_pos(1,:) = pos - floor(sub_target_sz/2);
            sub_init_pos(2,:) = pos + [-floor(sub_target_sz(1)/2), floor(sub_target_sz(2)/2)];
            sub_init_pos(3,:) = pos + [floor(sub_target_sz(1)/2), -floor(sub_target_sz(2)/2)];
            sub_init_pos(4,:) = pos + floor(sub_target_sz/2);
            old_sub_pos = sub_init_pos;
            for num = 1:subw_num
                % Extract training sample image region 
                pixels = get_pixels(im,sub_init_pos(num,:),round(sz*currentScaleFactor),sz);
                % Extract features and do windowing 
                xf = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params),cos_window));
                model_xf{num} = ((1 - learning_rate) * model_xf{num}) + (learning_rate * xf);
                % ADMM intialization
                g_f = single(zeros(size(xf))); 
                h_f = g_f;
                l_f = g_f;
                mu    = 1;
                betha = 10;
                mumax = 10000;
                i = 1;
                T = prod(use_sz/2); 
                S_xx = sum(conj(model_xf{num}) .* model_xf{num}, 3); 
                params.admm_iterations = 2; 
                % ADMM iteration
                while (i <= params.admm_iterations)
                    % Solve for G- please refer to the paper for more details
                    B = S_xx + (T * mu);
                    S_lx = sum(conj(model_xf{num}) .* l_f, 3); %xl
                    S_hx = sum(conj(model_xf{num}) .* h_f, 3); %xh
                    g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf{num})) - ((1/mu) * l_f) + h_f) - ...
                       bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf{num}, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf{num}, S_lx)) + (bsxfun(@times, model_xf{num}, S_hx))), B);
                    % Solve for H
                    h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f); 
                    [sx,sy,h] = get_subwindow_no_window(h, floor((use_sz)/2) , small_filter_sz);
                    t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
                    t(sx,sy,:) = h; 
                    h_f = fft2(t);  
                    % Update L 
                    l_f = l_f + (mu * (g_f - h_f));
                    % Update mu- betha = 10.
                    mu = min(betha * mu, mumax); % 
                    i = i+1;
                end 
            % Save the g_f of sub-regions
            sub_g_f(:,:,:,num) = g_f;  
            end 
        %%    
        else
            for num = 1:subw_num
                for scale_ind = 1:nScales
                    multires_pixel_template(:,:,:,scale_ind) = ...
                       get_pixels(im, sub_init_pos(num,:), round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);   
                end 
                xtf = fft2(bsxfun(@times,get_features(multires_pixel_template,features,global_feat_params),cos_window));
                responsef = permute(sum(bsxfun(@times, conj(sub_g_f(:,:,:,num)), xtf), 3), [1 2 4 3]);
                % If we undersampled features, we want to interpolate the response so it has the same size as the image patch
                if interpolate_response == 2
                    % Use dynamic interp size
                    interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
                end
                responsef_padded = resizeDFT2(responsef, interp_sz); 
                % Response in the spatial domain 
                response = ifft2(responsef_padded, 'symmetric');
                [disp_row, disp_col] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz);  
                
                sub_pos(:,:,num) = round(bsxfun(@plus, [disp_row' disp_col']*currentScaleFactor*featureRatio, sub_init_pos(num,:)));           
                %get APCE
                [apce(:,num),peak_loc] = get_PMLR(response, use_sz, nScales);
                sub_beta(:,num) = gamma.*apce(:,num);
                %weighting every response map
                for n = 1:nScales
                    modify_response(:,:,n) = bsxfun(@times, fftshift(response(:,:,n)), cos_mask(cos_window, peak_loc(n,:)));%
                end
                modify_response = bsxfun(@times,modify_response, permute(sub_beta(:,num), [3 2 1]));
                sub_response(:,:,:,num) = interp_res(modify_response, nScales, featureRatio);
            end   
            % response map fusion
            corner_ur  = sub_init_pos(4,:);
            corner_tl  = sub_init_pos(1,:);
            distance = abs(corner_ur - corner_tl);
            im_res = zeros(corner_ur(1)-corner_tl(1)+use_sz(1)*featureRatio, corner_ur(2)-corner_tl(2)+use_sz(2)*featureRatio ,nScales);
            Vp_2= zeros(corner_ur(1)-corner_tl(1)+use_sz(1)*featureRatio, corner_ur(2)-corner_tl(2)+use_sz(2)*featureRatio ,nScales);
            Vp_3= zeros(corner_ur(1)-corner_tl(1)+use_sz(1)*featureRatio, corner_ur(2)-corner_tl(2)+use_sz(2)*featureRatio ,nScales);
            Vp_4= zeros(corner_ur(1)-corner_tl(1)+use_sz(1)*featureRatio, corner_ur(2)-corner_tl(2)+use_sz(2)*featureRatio ,nScales);
            im_res(1:use_sz(1)*featureRatio, 1:use_sz(2)*featureRatio,:)= im_res(1:use_sz(1)*featureRatio, 1:use_sz(2)*featureRatio,:) + sub_response(:,:,:,1);
            Vp_1= im_res;
            im_res(1:use_sz(1)*featureRatio, distance(2)+1:end,:)=im_res(1:use_sz(1)*featureRatio, distance(2)+1:end,:) + sub_response(:,:,:,2);
            im_res(distance(1)+1:end, 1:use_sz(2)*featureRatio,:)= im_res(distance(1)+1:end, 1:use_sz(2)*featureRatio,:) + sub_response(:,:,:,3);
            im_res(distance(1)+1:end, distance(2)+1:end,:) = im_res(distance(1)+1:end, distance(2)+1:end,:) + sub_response(:,:,:,4);
            Vp_2(1:use_sz(1)*featureRatio, distance(2)+1:end,:)= sub_response(:,:,:,2);
            Vp_3(distance(1)+1:end, 1:use_sz(2)*featureRatio,:)= sub_response(:,:,:,3);
            Vp_4(distance(1)+1:end, distance(2)+1:end,:) = sub_response(:,:,:,4);
            Vp = im_res; 
            [~,sc_index]= max(sum(sum(Vp,1),2));
            % Initial translation estimation
            shift_vector = permute(sub_pos(sc_index,:,:),[3,2,1]) - old_sub_pos;
            pre_pos = sum(bsxfun(@times,shift_vector,(sub_beta(sc_index,:)./sum(sub_beta(sc_index,:)))'));
            
            pre_scale = currentScaleFactor_half;
            currentScaleFactor = currentScaleFactor_half;
%              target_sz = base_target_sz*currentScaleFactor;
             % Update initial location and scale changes
             p = [size(Vp, 2)/2+pre_pos(2) size(Vp, 1)/2+pre_pos(1) target_sz(2)*pre_scale target_sz(1)*pre_scale 0];
             %p = [size(Vp, 2)/2+pre_pos(2) size(Vp, 1)/2+pre_pos(1) target_sz(2) target_sz(1) 0];
             % Bayesian Inference framework
             param.est = affparam2mat([p(1), p(2), 1, p(5), p(4)/p(3), 0]);
             param.param = repmat(affparam2geom(param.est(:)), [1,n_sample])+ randn(6,n_sample).*repmat(affsig(:),[1,n_sample]);
             % Observation model calculation
             [res,~] = observation_score(Vp(:,:,sc_index),Vp_1(:,:,sc_index) ,Vp_2(:,:,sc_index) ,Vp_3(:,:,sc_index) ,Vp_4(:,:,sc_index) ,affparam2mat(param.param), target_sz*pre_scale, 1);

             %Update the final location 
             pos = pos + round(res.pos-[size(Vp, 1)/2 size(Vp, 2)/2]); 
             % output the result
             pos_result = pos;
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             pos_half = pos/2;
             pixels_half = get_pixels(im_pyramid,pos_half,round(sz_half*currentScaleFactor_half),sz_half);
             % extract features and do windowing
             xf_half = fft2(bsxfun(@times,get_features(pixels_half,features,global_feat_params_half),cos_window_half));
             model_xf_half = ((1 - half_worse_learningrate) * model_xf_half) + (half_worse_learningrate * xf_half);
             g_f_half = single(zeros(size(xf_half)));
             h_f_half = g_f_half;
             l_f_half = g_f_half;
             mu_half    = 1;
             betha_half = 10;
             mumax_half = 10000;
             i = 1;
             T_half = prod(use_sz_half);
             S_xx_half = sum(conj(model_xf_half) .* model_xf_half, 3);
             params.admm_iterations = 2;
             %ADMM
             while (i <= params.admm_iterations)
                %   solve for G- please refer to the paper for more details
                B_half = S_xx_half + (T_half * mu_half);
                S_lx_half = sum(conj(model_xf_half) .* l_f_half, 3);
                S_hx_half = sum(conj(model_xf_half) .* h_f_half, 3);
                g_f_half = (((1/(T_half*mu_half)) * bsxfun(@times, yf_half, model_xf_half)) - ((1/mu_half) * l_f_half) + h_f_half) - ...
                    bsxfun(@rdivide,(((1/(T_half*mu_half)) * bsxfun(@times, model_xf_half, (S_xx_half .* yf_half))) - ((1/mu_half) * bsxfun(@times, model_xf_half, S_lx_half)) + (bsxfun(@times, model_xf_half, S_hx_half))), B_half);
                %   solve for H
                h_half = (T_half/((mu_half*T_half)+ params.admm_lambda))* ifft2((mu_half*g_f_half) + l_f_half);
                [sx_half,sy_half,h_half] = get_subwindow_no_window(h_half, floor(use_sz_half/2) , small_filter_sz_half);
                t_half = single(zeros(use_sz_half(1), use_sz_half(2), size(h_half,3)));
                t_half(sx_half,sy_half,:) = h_half;
                h_f_half = fft2(t_half);
                %   update L
                l_f_half = l_f_half + (mu_half * (g_f_half - h_f_half));
                %   update mu- betha = 10.
                mu_half = min(betha_half * mu_half, mumax_half);
                i = i+1;
             end 
             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             % updating the four part-based filters
             %target_sz = target_sz_result;
             sub_target_sz = target_sz/2;
             % Location of sub-regions
             sub_init_pos(1,:) = pos - floor(sub_target_sz/2);
             sub_init_pos(2,:) = pos + [-floor(sub_target_sz(1)/2), floor(sub_target_sz(2)/2)];
             sub_init_pos(3,:) = pos + [floor(sub_target_sz(1)/2), -floor(sub_target_sz(2)/2)];
             sub_init_pos(4,:) = pos + floor(sub_target_sz/2);
             old_sub_pos = sub_init_pos;
             for num = 1:subw_num
                % Extract training sample image region 
                pixels = get_pixels(im,sub_init_pos(num,:),round(sz*currentScaleFactor),sz);
                % Extract features and do windowing 
                xf = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params),cos_window));
                model_xf{num} = ((1 - learning_rate) * model_xf{num}) + (learning_rate * xf);
                % ADMM intialization
                g_f = single(zeros(size(xf))); 
                h_f = g_f;
                l_f = g_f;
                mu    = 1;
                betha = 10;
                mumax = 10000;
                i = 1;
                T = prod(use_sz/2); 
                S_xx = sum(conj(model_xf{num}) .* model_xf{num}, 3); 
                params.admm_iterations = 2; 
                % ADMM iteration
                while (i <= params.admm_iterations)
                    % Solve for G- please refer to the paper for more details
                    B = S_xx + (T * mu);
                    S_lx = sum(conj(model_xf{num}) .* l_f, 3); %xl
                    S_hx = sum(conj(model_xf{num}) .* h_f, 3); %xh
                    g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf{num})) - ((1/mu) * l_f) + h_f) - ...
                       bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf{num}, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf{num}, S_lx)) + (bsxfun(@times, model_xf{num}, S_hx))), B);
                    % Solve for H
                    h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f); 
                    [sx,sy,h] = get_subwindow_no_window(h, floor((use_sz)/2) , small_filter_sz);
                    t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
                    t(sx,sy,:) = h; 
                    h_f = fft2(t);  
                    % Update L 
                    l_f = l_f + (mu * (g_f - h_f));
                    % Update mu- betha = 10.
                    mu = min(betha * mu, mumax); % 
                    i = i+1;
                end 
            % Save the g_f of sub-regions
             sub_g_f(:,:,:,num) = g_f;  
             end
             %pos_half = pos/2;
        end  
    end    
    %% saving the result and visualisation
    %save position and calculate FPS
    time = time + toc();
    %save position and calculate FPS
    %rect_position(loop_frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    rect_position(loop_frame,:) = [pos_result([2,1]) - floor(target_sz_result([2,1])/2), target_sz_result([2,1])];
    % Visualization 
    if visualization == 1
%         rect_position_vis = [pos_result([2,1]) - target_sz_result([2,1])/2, target_sz_result([2,1])];
         rect_position_vis = [pos_result([2,1])/2 - target_sz_result([2,1])/4, target_sz_result([2,1])/2];
        %im_to_show = double(im)/255;
         im_to_show = double(im_pyramid)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            resp_sz = round(sz_half*currentScaleFactor_half*scaleFactors_half(scale_ind));
            xs = floor(old_pos_half(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(old_pos_half(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            sc_ind = floor((nScales - 1)/2) + 1;
            
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, fftshift(response_half(:,:,sc_ind))); colormap hsv;
            alpha(resp_handle, 0.2);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(20, 30, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            text(20, 60, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            
            hold off;
        end
        drawnow
    end 
    loop_frame = loop_frame + 1;
end

% Save tracking resutls
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
