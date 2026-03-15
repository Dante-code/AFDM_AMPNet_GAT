function [H_eff, heff_info] = buildHeff_DAF(N_blk, c1, c2, chan)
%BUILDHEFF_DAF 构造 AFDM DAF 域等效信道 H_eff (N_blk×N_blk)
%   采用矩阵乘法路径：
%     H = sum_i h_i * G_i * D_i * Pi^(l_i)
%     H_eff = A * H * A'
%   其中分数多普勒通过 D_i 中 nu_i = kp_i + kappa_i 注入。
%
%   输出:
%     H_eff: (N,N) 复数等效信道
%     heff_info: 主项/扩展项标注信息（用于后续数据标注）

    P  = chan.P;        % 路径条数
    lp = chan.lp(:);    % 每条路径的延迟索引（列向量）
    kp = chan.kp(:);    % 每条路径的整数多普勒索引（列向量）
    if isfield(chan, "kappa")
        kappa = chan.kappa(:); % 每条路径的分数多普勒
    else
        % 兼容旧数据：未提供分数多普勒时按 0 处理
        kappa = zeros(P, 1);
    end
    hp = chan.hp(:);    % 每条路径的复增益

    if isfield(chan, "kv")
        % kv 表示主项两侧保留的扩展宽度，强制为非负整数
        kv = max(0, round(double(chan.kv)));
    else
        % 兼容旧数据：未提供 kv 时按整数多普勒（仅主项）处理
        kv = 0;
    end

    N = N_blk;          % 单帧长度
    nn = (0:N-1)';      % 采样索引列向量
    lambda_c1 = exp(-1j * 2 * pi * c1 * nn.^2);  % c1 对应啁啾相位
    lambda_c2 = exp(-1j * 2 * pi * c2 * nn.^2);  % c2 对应啁啾相位
    F = dftmtx(N) / sqrt(N);                      % 归一化 DFT 矩阵
    A = diag(lambda_c2) * F * diag(lambda_c1);   % DAFT 变换矩阵

    [H_eff, ~, heff_info] = gen_Heff_fractional(N, hp, kp, kappa, lp, P, c1, kv, A);

    % 仅移除数值噪声，不破坏分数多普勒展开结构
    prune_tol = 1e-12;
    H_eff(abs(H_eff) < prune_tol) = 0;
end

function [Heff, H, heff_info] = gen_Heff_fractional(N, h, kp, kappa, l_t, P, c1, kv, A)
%GEN_HEFF_FRACTIONAL 用 H 域矩阵累加构造 Heff，并输出主/扩展项标注
    nu = kp + kappa;                              % 总多普勒（整数+分数）
    alpha = round(nu);                            % 总多普勒的整数近似部分
    loc_main = mod(alpha + round(2 * N * c1 * l_t), N); % 每条路径主项列偏移

    % 循环移位矩阵 Pi（与旧版保持一致）
    pi_row = [zeros(1, N - 1), 1];
    Pi = toeplitz([pi_row(1), fliplr(pi_row(2:end))], pi_row);

    H = complex(zeros(N, N));  % 时域等效信道矩阵
    n = (0:N-1)';              % 行索引（用于构造相位补偿）
    for i = 1:P
        hi = h(i);    % 第 i 条路径复增益
        li = l_t(i);  % 第 i 条路径延迟
        nui = nu(i);  % 第 i 条路径总多普勒

        % D_i：由多普勒引入的对角相位矩阵
        Di = diag(exp(-1j * 2 * pi * (nui / N) * (0:N-1)));

        % G_i：CPP 相位补偿矩阵
        temp = ones(N, 1);     % 默认相位为 1
        idx = n < li;          % 需要补偿的前 li 个位置
        temp(idx) = exp(-1j * 2 * pi * c1 * (N^2 - 2 * N * (li - n(idx))));
        Gi = diag(temp);

        % 按路径累加：h_i * G_i * D_i * Pi^{l_i}
        H = H + hi * Gi * Di * (Pi ^ li);
    end

    % 时域转 DAFT 域
    Heff = A * H * A';

    % --- 向量化构建主项/扩展项标注（无 p/q 双循环） ---
    p_idx = (0:N-1)';                % 所有行索引（0 基）
    main_mask_by_path = false(N, N, P); % 每条路径的主项掩码
    ext_mask_by_path = false(N, N, P);  % 每条路径的扩展项掩码
    main_q_by_path = zeros(N, P);    % 每条路径每一行对应的主项列索引（0 基）
    delta_set = (-kv:kv);            % 扩展偏移集合
    for i = 1:P
        % 主项列位置：q = (p + loc_i) mod N（0 基）
        q_main = mod(p_idx + loc_main(i), N);
        main_q_by_path(:, i) = q_main;

        % 将二维坐标 (行, 列) 映射为 MATLAB 线性索引，便于一次性赋值
        lin_main = sub2ind([N, N], (1:N)', q_main + 1);
        m = false(N, N);  % 第 i 条路径主项掩码
        m(lin_main) = true;
        main_mask_by_path(:, :, i) = m;

        e = false(N, N);  % 第 i 条路径扩展项掩码
        for d_idx = 1:numel(delta_set)
            delta = delta_set(d_idx);
            if delta == 0
                % 偏移为 0 已计入主项，不重复计入扩展项
                continue;
            end
            % 扩展项列位置：q = (q_main + delta) mod N
            q_ext = mod(q_main + delta, N);
            lin_ext = sub2ind([N, N], (1:N)', q_ext + 1);
            e(lin_ext) = true;
        end
        ext_mask_by_path(:, :, i) = e;
    end

    % 将按路径掩码聚合成全局掩码（任一路径命中即为 true）
    main_mask = any(main_mask_by_path, 3);
    ext_mask = any(ext_mask_by_path, 3);

    % 标注信息统一打包，供数据集脚本保存和 Python 端使用
    heff_info = struct();
    heff_info.kv = kv;
    heff_info.nu = nu;
    heff_info.alpha = alpha;
    heff_info.loc_main = loc_main;
    heff_info.main_q_by_path = main_q_by_path;
    heff_info.main_mask = main_mask;
    heff_info.ext_mask = ext_mask;
    heff_info.main_mask_by_path = main_mask_by_path;
    heff_info.ext_mask_by_path = ext_mask_by_path;
    heff_info.delta_offsets = delta_set;
end
