function chan = generateChannel(N, P, l_max, k_max, kv)
%GENERATECHANNEL 按多径模型生成 P 条路径的 (h_p, l_p, k_p, kappa_p)
%   N: 块大小，供 AFDM buildHeff_DAF 使用。
%   kv: 分数多普勒展开半宽（透传到 buildHeff_DAF）

    if nargin < 5
        kv = 0;
    end

    % 整数延迟 l_p ∈ {0,...,l_max}
    lp = randi([0, l_max], P, 1);

    % 整数多普勒 k_p ∈ {-k_max,...,k_max}
    kp = randi([-k_max, k_max], P, 1);

    % 分数多普勒 kappa，满足 nu = kp + kappa ∈ [-k_max, k_max]
    kappa = zeros(P, 1);
    for i = 1:P
        low_i = max(-0.5, -k_max - kp(i));
        high_i = min(0.5, k_max - kp(i));
        if low_i > high_i
            % 理论上不会发生，兜底置零以保证数值稳定
            kappa(i) = 0;
        else
            kappa(i) = low_i + (high_i - low_i) * rand();
        end
    end

    % 功率谱 σ_p^2 ∝ exp(-0.1 l_p)，再归一化（贴论文仿真设置）
    % pow_unnorm = exp(-0.1 * double(lp));
    % pow_norm   = pow_unnorm / sum(pow_unnorm);

    % 复高斯增益 h_p ~ CN(0, σ_p^2)
    hp = (randn(P,1) + 1j*randn(P,1)) .* sqrt(1 / 2 / P);

    chan.N     = N;
    chan.P     = P;
    chan.lp    = lp;
    chan.kp    = kp;
    chan.kappa = kappa;
    chan.kv    = kv;
    chan.hp    = hp;
end