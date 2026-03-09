function [x_daf, y_daf, H_eff, chan, sigma2] = simulateAFDM(N, c1, c2, P, l_max, k_max, SNR_dB, QAM_order)
%simulateAFDM 一帧 AFDM DAF 域仿真（仅整数延迟/多普勒，无分数多普勒）
%   N: 块大小（DAF 域子载波数），1D
%   使用 buildHeff_DAF，适合 AMP-GNN 轻量实验。
%
%   输出含 sigma2 便于 Python 端使用。
    %% 1. 生成 DAF 域发射符号 x_daf (QAM)
    M_qam = QAM_order;
    x_idx = randi([0, M_qam-1], N, 1);
    x_daf  = qammod(x_idx, M_qam, 'UnitAveragePower', true);

    %% 2. 生成多径通道参数（kappa 被 buildHeff_DAF 忽略）
    chan = generateChannel(N, P, l_max, k_max);

    %% 3. 构造 DAF 域等效信道 H_eff
    H_eff = buildHeff_DAF(N, c1, c2, chan);

    %% 4. 通过 H_eff 产生接收 DAF 符号，并加噪声
    xvec = x_daf;

    Es = mean(abs(xvec).^2);
    SNR_lin = 10.^(SNR_dB/10);
    sigma2  = Es / SNR_lin;

    noise = sqrt(sigma2/2) * (randn(N,1) + 1j*randn(N,1));
    yvec = H_eff * xvec + noise;
    y_daf = yvec;
end
