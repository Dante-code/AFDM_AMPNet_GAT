function chan = generateChannel(N, P, l_max, k_max)
%GENERATECHANNEL 按多径模型生成 P 条路径的 (h_p, l_p, k_p, kappa_p)
%   N: 块大小，供 AFDM buildHeff_DAF 使用。

    % 整数延迟 l_p ∈ {0,...,l_max}
    lp = randi([0, l_max], P, 1);

    % 整数多普勒 k_p ∈ {-k_max,...,k_max}
    kp = randi([-k_max, k_max], P, 1);

    % 分数多普勒 κ_p ∈ [-1/2, 1/2]
    kappa = -0.5 + rand(P, 1);

    % 功率谱 σ_p^2 ∝ exp(-0.1 l_p)，再归一化（贴论文仿真设置）
    % pow_unnorm = exp(-0.1 * double(lp));
    % pow_norm   = pow_unnorm / sum(pow_unnorm);

    % 复高斯增益 h_p ~ CN(0, σ_p^2)
    hp = (randn(P,1) + 1j*randn(P,1)) .* sqrt(P / 2);

    chan.N     = N;
    chan.P     = P;
    chan.lp    = lp;
    chan.kp    = kp;
    chan.kappa = kappa;
    chan.hp    = hp;
end