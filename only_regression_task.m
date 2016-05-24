function only_regression_task()    
    caffe_version          = 'caffe';
    gpu_id                 = 1;
    active_caffe_mex(gpu_id, caffe_version);   
    
    dir='/media/cgv841/Code/tydu/regression_task/';
    solver_def_file=[dir 'solver.prototxt'];
    net_file=[dir 'vgg16.caffemodel'];
    
    load([dir 'ibug_regression_dataset.mat'])    
    load([dir 'ibug_regression_mean_face.mat'])
    
    caffe_log_file_base = fullfile(dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    caffe_solver = caffe.Solver(solver_def_file);
    caffe_solver.net.copy_from(net_file);

    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(dir, 'log'));
    log_file = fullfile(dir, 'log', ['train_', timestamp, '.txt']);
    diary(log_file);
    
    % set random seed
    caffe.set_random_seed(6);
    
    % set gpu/cpu
    caffe.set_mode_gpu();
    %caffe.set_mode_cpu();
    
%% making tran/val data
%          % fix validation data
%         shuffled_inds_val = generate_random_minibatch([], image_roidb_val, conf.ims_per_batch);
%         shuffled_inds_val = shuffled_inds_val(randperm(length(shuffled_inds_val), opts.val_iters));
%     end      
%% training
    shuffled_inds = [];
    train_results = [];  
    %val_results = [];  
    iter_ = caffe_solver.iter();
    max_iter = caffe_solver.max_iter();    
    
    while (iter_ < max_iter)
        caffe_solver.net.set_phase('train');
        % generate minibatch training data
        [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, ibug_regression_dataset, 1);
        
        [im_blob, ~] = get_image_blob(mean_image, ibug_regression_dataset(sub_inds));

        im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
        im_blob = single(permute(im_blob, [2, 1, 3, 4]));                             
        landmark_label_blob = ibug_regression_dataset(sub_inds).landmark';
        landmark_label_blob = single(permute(landmark_label_blob, [3, 4, 2, 1]));
                
        net_inputs = {im_blob, landmark_label_blob};
        
        caffe_solver.net.reshape_as_input(net_inputs);

        % one iter SGD update
        caffe_solver.net.set_input_data(net_inputs);
        caffe_solver.step(1);
        
        rst = caffe_solver.net.get_output();
        train_results = parse_rst(train_results, rst);
            
        % do valdiation per val_interval iterations
%         if ~mod(iter_, opts.val_interval) 
%             if opts.do_val
%                 caffe_solver.net.set_phase('test');                
%                 for i = 1:length(shuffled_inds_val)
%                     sub_db_inds = shuffled_inds_val{i};
%                     [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob] = ...
%                         fast_rcnn_get_minibatch(conf, image_roidb_val(sub_db_inds));
% 
%                     % Reshape net's input blobs
%                     net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};
%                     caffe_solver.net.reshape_as_input(net_inputs);
%                     
%                     caffe_solver.net.forward(net_inputs);
%                     
%                     rst = caffe_solver.net.get_output();
%                      val_results = parse_rst(val_results, rst);
%                 end
%             end
%             show_state(iter_, train_results, val_results);
%             train_results = [];
%             val_results = [];
%             diary; diary; % flush diary
%         end
        if ~mod(iter_,2000)
            show_state(iter_, train_results, val_results);
        end
            % snapshot
        if ~mod(iter_, 20000)
            model_path = fullfile(dir, sprintf('iter_%d', iter_));
            caffe_solver.net.save(model_path);
            fprintf('Saved as %s\n', model_path);
        end
        
        iter_ = caffe_solver.iter();
    end
    
    model_path = fullfile(dir, 'final');
    caffe_solver.net.save(model_path);
    diary off;
    caffe.reset_all(); 
end

function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, final_image, ims_per_batch)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        % make sure each minibatch, only has horizontal images or vertical
        % images, to save gpu memory
        shuffled_inds = find(ones(size(final_image,2),1));
        ind = randperm(size(final_image,2));
        shuffled_inds = shuffled_inds(ind);
        shuffled_inds = num2cell(shuffled_inds, 2);
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end

function [im_blob, im_scales] = get_image_blob(image_means, images)
    
    num_images = length(images);
    processed_ims = cell(num_images, 1);
    im_scales = nan(num_images, 1);
    for i = 1:num_images
        im=images(i).image; 
        target_size = 224;        
        [im, im_scale] = prep_im_for_blob(im, image_means, target_size, 1000);
        
        im_scales(i) = im_scale;
        processed_ims{i} = im; 
    end
    
    im_blob = im_list_to_blob(processed_ims);
end

function show_state(iter, train_results, val_results)
    fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
    %fprintf('Training : error %.3g, loss (cls %.3g, reg %.3g)\n', ...
    fprintf('Training : loss (ldm %.3g)\n', mean(train_results.loss_landmark.data)); 
    if exist('val_results', 'var') && ~isempty(val_results)
        fprintf('Testing  : error %.3g, loss (cls %.3g, reg %.3g)\n', ...
            1 - mean(val_results.accuarcy.data), ...
            mean(val_results.loss_cls.data), ...
            mean(val_results.loss_bbox.data));
    end
end