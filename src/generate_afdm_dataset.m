% generate_afdm_dataset.m
% Generate AFDM dataset for one split per run to reduce MATLAB memory usage.
% Output:
%   - afdm_train.mat OR afdm_val.mat OR afdm_test.mat
%   - dataset_meta.yaml (shared metadata for Python config validation)

clear;

%% ===== User Config =====
split_mode = "test";  % "train" | "val" | "test"
output_dir = "./data/";      % usually src/
train_mat_name = "afdm_train_snr_14db";
test_mat_name = "afdm_test_snr_14db";
val_mat_name = "afdm_val_snr_14db";
dataset_meta_yaml_name = "dataset_meta";
N = 128;
P = 4;
l_max = 2;
k_max = 3;
SNR_dB = 14;
QAM_order = 4;  % QPSK
xi_v = 1;
c1 = (2 * (k_max + xi_v) + 1) / (2 * N);
c2 = pi / 50;

n_train = 30000;
n_val = 30000;
n_test = 30000;
SNR_test = [10, 12, 14, 16];
%% =======================

if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

split_mode = lower(string(split_mode));
if split_mode ~= "train" && split_mode ~= "val" && split_mode ~= "test"
    error("split_mode must be one of: train, val, test");
end

fprintf("Generating AFDM %s split...\n", split_mode);
fprintf("  N=%d, P=%d, QAM=%d\n", N, P, QAM_order);

if split_mode == "train"
    n_cur = n_train;
    x_arr = complex(zeros(n_cur, N), zeros(n_cur, N));
    y_arr = complex(zeros(n_cur, N), zeros(n_cur, N));
    H_arr = complex(zeros(n_cur, N, N), zeros(n_cur, N, N));
    sigma2_arr = zeros(n_cur, 1);

    for i = 1:n_cur
        [x_daf, y_daf, H_eff, ~, sigma2] = ...
            simulateAFDM(N, c1, c2, P, l_max, k_max, SNR_dB, QAM_order);
        x_arr(i, :) = x_daf(:).';
        y_arr(i, :) = y_daf(:).';
        H_arr(i, :, :) = H_eff;
        sigma2_arr(i) = sigma2;
        if mod(i, 1000) == 0
            fprintf("  train: %d/%d\n", i, n_cur);
        end
    end

    out_path = fullfile(output_dir, strcat(train_mat_name,".mat"));
    x_daf_train_arr = x_arr; %#ok<NASGU>
    y_daf_train_arr = y_arr; %#ok<NASGU>
    H_eff_train_arr = H_arr; %#ok<NASGU>
    sigma2_train = sigma2_arr; %#ok<NASGU>
    save(out_path, ...
        "x_daf_train_arr", "y_daf_train_arr", "H_eff_train_arr", "sigma2_train", ...
        "N", "P", "QAM_order", "SNR_dB", "n_train", "l_max", "k_max", "c1", "c2", "-v7.3");
end

if split_mode == "val"
    n_cur = n_val;
    x_arr = complex(zeros(n_cur, N), zeros(n_cur, N));
    y_arr = complex(zeros(n_cur, N), zeros(n_cur, N));
    H_arr = complex(zeros(n_cur, N, N), zeros(n_cur, N, N));
    sigma2_arr = zeros(n_cur, 1);

    for i = 1:n_cur
        [x_daf, y_daf, H_eff, ~, sigma2] = ...
            simulateAFDM(N, c1, c2, P, l_max, k_max, SNR_dB, QAM_order);
        x_arr(i, :) = x_daf(:).';
        y_arr(i, :) = y_daf(:).';
        H_arr(i, :, :) = H_eff;
        sigma2_arr(i) = sigma2;
        if mod(i, 500) == 0
            fprintf("  val: %d/%d\n", i, n_cur);
        end
    end

    out_path = fullfile(output_dir, strcat(val_mat_name,".mat"));
    x_daf_val_arr = x_arr; %#ok<NASGU>
    y_daf_val_arr = y_arr; %#ok<NASGU>
    H_eff_val_arr = H_arr; %#ok<NASGU>
    sigma2_val = sigma2_arr; %#ok<NASGU>
    save(out_path, ...
        "x_daf_val_arr", "y_daf_val_arr", "H_eff_val_arr", "sigma2_val", ...
        "N", "P", "QAM_order", "SNR_dB", "n_val", "l_max", "k_max", "c1", "c2", "-v7.3");
end

if split_mode == "test"
    n_cur = n_test;
    x_arr = complex(zeros(n_cur, N), zeros(n_cur, N));
    y_arr = complex(zeros(n_cur, N), zeros(n_cur, N));
    H_arr = complex(zeros(n_cur, N, N), zeros(n_cur, N, N));
    sigma2_arr = zeros(n_cur, 1);
    SNR_test_vec = zeros(n_cur, 1); %#ok<NASGU>

    n_test_per_snr = ceil(n_test / length(SNR_test));
    idx = 1;
    for s = 1:length(SNR_test)
        for i = 1:n_test_per_snr
            if idx > n_cur
                break;
            end
            [x_daf, y_daf, H_eff, ~, sigma2] = ...
                simulateAFDM(N, c1, c2, P, l_max, k_max, SNR_test(s), QAM_order);
            x_arr(idx, :) = x_daf(:).';
            y_arr(idx, :) = y_daf(:).';
            H_arr(idx, :, :) = H_eff;
            sigma2_arr(idx) = sigma2;
            SNR_test_vec(idx) = SNR_test(s);
            idx = idx + 1;
        end
        fprintf("  test SNR=%d dB done\n", SNR_test(s));
    end

    out_path = fullfile(output_dir, strcat(test_mat_name,".mat"));
    x_daf_test_arr = x_arr; %#ok<NASGU>
    y_daf_test_arr = y_arr; %#ok<NASGU>
    H_eff_test_arr = H_arr; %#ok<NASGU>
    sigma2_test = sigma2_arr; %#ok<NASGU>
    save(out_path, ...
        "x_daf_test_arr", "y_daf_test_arr", "H_eff_test_arr", "sigma2_test", "SNR_test_vec", ...
        "N", "P", "QAM_order", "n_test", "l_max", "k_max", "c1", "c2", "-v7.3");
end

meta_path = fullfile(output_dir, strcat(dataset_meta_yaml_name,".yaml"));
write_dataset_meta_yaml(meta_path, N, P, QAM_order, l_max, k_max, c1, c2, ...
    SNR_dB, n_train, n_val, n_test, SNR_test, ...
    struct("train_mat_name", train_mat_name, "val_mat_name", val_mat_name, "test_mat_name", test_mat_name));

fprintf("Saved split file for %s in %s\n", split_mode, output_dir);
fprintf("Saved metadata: %s\n", meta_path);


function write_dataset_meta_yaml(path, N, P, QAM_order, l_max, k_max, c1, c2, ...
        SNR_dB, n_train, n_val, n_test, SNR_test, opts)
    
    if nargin < 14, opts = struct; end
    train_mat_name = getopt(opts, "train_mat_name", "afdm_train");
    val_mat_name = getopt(opts, "val_mat_name", "afdm_val");
    test_mat_name = getopt(opts, "test_mat_name", "afdm_test");

    
    fid = fopen(path, "w");
    if fid < 0
        error("Cannot open metadata file for writing: %s", path);
    end
    cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

    ts = string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));

    fprintf(fid, "dataset:\n");
    fprintf(fid, "  version: 1\n");
    fprintf(fid, "  generated_by: generate_afdm_dataset.m\n");
    fprintf(fid, "  updated_at: \'%s\'\n", ts);
    fprintf(fid, "  common:\n");
    fprintf(fid, "    N: %d\n", N);
    fprintf(fid, "    P: %d\n", P);
    fprintf(fid, "    QAM_order: %d\n", QAM_order);
    fprintf(fid, "    l_max: %d\n", l_max);
    fprintf(fid, "    k_max: %d\n", k_max);
    fprintf(fid, "    c1: %.16g\n", c1);
    fprintf(fid, "    c2: %.16g\n", c2);
    fprintf(fid, "    snr_train_db: %.16g\n", SNR_dB);
    fprintf(fid, "  splits:\n");
    fprintf(fid, "    train:\n");
    fprintf(fid, "      file: %s.mat\n", train_mat_name);
    fprintf(fid, "      count: %d\n", n_train);
    fprintf(fid, "    val:\n");
    fprintf(fid, "      file: %s.mat\n", val_mat_name);
    fprintf(fid, "      count: %d\n", n_val);
    fprintf(fid, "    test:\n");
    fprintf(fid, "      file: %s.mat\n", test_mat_name);
    fprintf(fid, "      count: %d\n", n_test);
    fprintf(fid, "      snr_test_db: [");
    for i = 1:length(SNR_test)
        if i > 1
            fprintf(fid, ", ");
        end
        fprintf(fid, "%.16g", SNR_test(i));
    end
    fprintf(fid, "]\n");
end

function v = getopt(s, name, default)
    if isfield(s, name), v = s.(name); else, v = default; end
end