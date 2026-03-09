function H_eff = buildHeff_DAF(N_blk, c1, c2, chan)
%BUILDHEFF_DAF 构造 AFDM DAF 域等效信道 H_eff (N_blk×N_blk)
%   H_eff = A * H * A'，A 为 DAFT 矩阵，H 为时域信道矩阵。
%   向量化后： ȳ = H_eff x̄ + w̄

    P     = chan.P;
    lp    = chan.lp(:);    % 延迟索引（采样）
    kp    = chan.kp(:);    % 多普勒索引
    kappa = chan.kappa(:); % 分数多普勒（本实现中未用）
    hp    = chan.hp(:);    % 路径增益

    N = N_blk;
    nn = (0:N-1)';
    lambda_c1 = exp(-1j*2*pi*c1*nn.^2);
    lambda_c2 = exp(-1j*2*pi*c2*nn.^2);


    F = dftmtx(N)/sqrt(N);
    A = diag(lambda_c2)*F*diag(lambda_c1);
    [H_eff, ~] = gen_Heff(N, hp, kp, lp, P, c1, A);
    H_eff(abs(H_eff) < 1e-2) = 0;
end

function [Heff, H] = gen_Heff(N, h, f_d, l_t, P, c1, A)   
    % 生成循环移位矩阵
    Pi = [zeros(1, N-1) 1];
    Pi = toeplitz([Pi(1) fliplr(Pi(2:end))], Pi);
    H = zeros(N);  % 初始化信道矩阵
    % 遍历每条路径，生成信道矩阵
    for i = 1:P
        hi = h(i);        % 第i条路径的复增益
        li = l_t(i);      % 第i条路径的延迟
        fi = f_d(i)/N;      % 第i条路径的多普勒频移
        % 生成多普勒频移矩阵
        Di = diag(exp(-1j*2*pi*fi*(0:N-1)));
        % 生成延迟相关的相位调整矩阵
        temp = ones(N, 1);
        for n = 0:N-1
            if n < li
                % 对于小于延迟的样本，应用相位调整
                temp(n+1) = exp(-1j*2*pi*c1*(N^2 - 2*N*(li-n)));
            else
                % 对于大于等于延迟的样本，相位不变
                temp(n+1) = 1;
            end
        end
        Gi = diag(temp);
        % 累加各路径的贡献到信道矩阵
        H = H + hi * Gi * Di * Pi^li;
    end
    Heff = A*H*A';
end
