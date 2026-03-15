# Affine Frequency Division Multiplexing for Next Generation Wireless Communications

**Ali Bemani**, *Member, IEEE*, **Nassar Ksairi**, *Senior Member, IEEE*, and **Marios Kountouris**, *Fellow, IEEE*

---

> **Abstract**— Affine Frequency Division Multiplexing (AFDM), a new chirp-based multicarrier waveform for high mobility communications, is introduced here. AFDM is based on discrete affine Fourier transform (DAFT), a generalization of discrete Fourier transform, which is characterized by two parameters that can be adapted based on the Doppler spread of doubly dispersive channels. First, we derive the explicit input-output relation in the DAFT domain showing the effect of AFDM parameters in the input-output relation. Second, we show how the DAFT parameters underlying AFDM have to be set so that the resulting DAFT domain impulse response conveys a full delay-Doppler representation of the channel. Then, we show analytically that AFDM can achieve the optimal diversity order in doubly dispersive channels, where optimal diversity order refers to the number of multipath components separable in either the delay or the Doppler domain, due to its full delay-Doppler representation. Furthermore, we present a low complexity detection method taking advantage of zero-padding. We also propose an embedded pilot-aided channel estimation scheme for AFDM, in which both channel estimation and data detection are performed within the same AFDM frame. Finally, simulations corroborate the validity of our analytical results and show the significant performance gains of AFDM over state-of-the-art multicarrier schemes in high mobility scenarios.

> **Index Terms**— Affine frequency division multiplexing, affine Fourier transform, chirp modulation, linear time-varying channels, doubly dispersive channels, high mobility communications.

---

## I. Introduction

Next generation wireless systems and standards (beyond 5G/6G) are expected to support a wide spectrum of services, including reliable communication in high mobility scenarios (e.g., V2X communications, flying vehicles, and high-speed rail systems) and in extremely high-frequency (EHF) bands. Current systems are based on Orthogonal Frequency Division Multiplexing (OFDM), a widely used multicarrier scheme that achieves near optimal performance in time-invariant frequency selective channels. Nevertheless, in time-varying channels (also referred to as doubly dispersive or doubly selective channels), the performance of OFDM drastically decreases. This is mainly due to large Doppler frequency shifts and the loss of orthogonality among subcarriers, resulting in inter-carrier interference (ICI). This calls for new modulation techniques and waveforms, which are able to cope with various challenging requirements and show robustness in high mobility scenarios.

One approach to compensate for fast variations in LTV channels is to shorten the OFDM symbol duration so that the channel variations over each symbol duration become negligible [3]. However, due to cyclic prefix (CP), this approach significantly reduces the spectral efficiency. In theory, the optimal approach to cope with time-varying multipath channels is to transmit information symbols leveraging an orthogonal eigenfunction decomposition of the channel and then project the received signal over the same set of orthogonal eigenfunctions at the receiver. In linear time-invariant (LTI) systems, complex exponentials are known to be eigenfunctions of the channel and can be obtained via the Fourier transform (FT). However, finding an orthonormal basis for general LTV channels is not trivial and polynomial phase models that generalize complex exponentials are often used as alternative bases.

Since this optimal approach presents significant challenges both in terms of conceptual and computational complexity, using chirps, i.e., complex exponentials with linearly varying instantaneous frequencies, appears to be a promising alternative. The use of chirps for communication and sensing purposes has a long history. S. Darlington in 1947 proposed the chirp technique for pulsed radar systems with long-range performance and high-range resolution [4]. The term "chirp" was apparently first employed by B. M Oliver in an internal Bell Laboratories Memorandum "Not with a bang, but a Chirp". In [5], an experimental communication system employing chirp modulation in the high frequency band for air-ground communication is presented. Since chirped waveforms are of spread-spectrum, they can also provide security and robustness in several scenarios, including military, underwater and aerospace communications [6], [7], [8]. Chirps are specified in the IEEE 802.15.4a standard as chirp spread spectrum (CSS) to meet the requirement of FCC on the radiation power spectral mask for the unlicensed UWB systems [9].

Using a frequency-varying basis for a multicarrier transmission scheme over time-varying channels is first introduced in [10]. In this work, an orthonormal basis formed by chirps are generated using Fractional Fourier Transform (FrFT). The scheme is presented in a continuous-time setting, whereas the approximation used for making the continuous-time FrFT discrete leads to imperfect orthogonality among chirp subcarriers and hence to performance degradation. A multicarrier technique based on Affine Fourier Transform (AFT), which is a generalization of the Fourier and fractional Fourier transform, is proposed in [11]. The resulting multicarrier waveform therein is referred to as DAFT-OFDM in the sequel where DAFT stands for Discrete AFT. It is equivalent to OFDM with reduced ICI on doubly dispersive channels and is shown to achieve low diversity order. Moreover, the delays and the Doppler shifts of channel paths are required at the transmitter in order to tune the DAFT-OFDM parameters. In [12], a general interference analysis of the DAFT-OFDM system is provided and the optimal parameters are obtained in closed form, followed by the analysis of the effects of synchronization errors and the optimal symbol period. Another scheme that is proposed for communication over time-dispersive channels is Orthogonal Chirp Division Multiplexing (OCDM) [13], which is based on the discrete Fresnel transform - a special case of DAFT. OCDM is shown to perform better than uncoded OFDM in LTI and LTV channels [14]. However, in LTI channels, OCDM can achieve diversity order of one for very large signal to noise ratio (SNR), whereas in general LTV channels, it cannot achieve the optimal diversity order since its diversity order depends on the delay-Doppler profile of the channel.

In addition to chirp-based modulation, several waveforms have been proposed to provide improved performance compared to OFDM in terms of carrier frequency offset (CFO) sensitivity, peak to average power ratio (PAPR), and out-of-band emissions (OOBE). Discrete Fourier transform spread OFDM (DFT-s-OFDM), also known as Single Carrier-Frequency Division Multiple Access (SC-FDMA), has been proposed in [15], which spreads symbol energy equally over all subcarriers to reduce PAPR by precoding data symbols using a DFT. Generalized Frequency Division Multiplexing (GFDM) [16], [17] is a multicarrier modulation based on a circular pulse shaping filter, aiming to reduce the OOBE. Nevertheless, DFT-s-OFDM and GFDM are sensitive to CFO due to Doppler spread. To deal with the Doppler spread, Orthogonal Time Frequency Space (OTFS) modulation has recently been proposed for high mobility communications [18], [19]. OTFS is a two-dimensional (2D) modulation technique that spreads the information symbols over the delay-Doppler domain. OTFS has been shown to outperform previously proposed waveforms in both frequency selective and doubly selective channels [20]. Therefore, in this paper, the performance of the proposed AFDM is compared with OTFS. The results in [21] show that the OTFS diversity order without channel coding is one and with a phase rotation scheme using transcendental numbers can be made equal to the optimal diversity order. The idea of embedding pilots along with the data symbols in the delay-Doppler domain has been proposed in [22]. Although no separate transmission for the pilot symbols is needed, OTFS suffers from excessive pilot overhead due to its 2D structure as each pilot symbol should be separated from the data symbols.

In this paper, we propose a novel multicarrier scheme called Affine Frequency Division Multiplexing (AFDM), which is a DAFT-based waveform using multiple orthogonal information-bearing chirp signals. The key idea is to multiplex information symbols in the DAFT domain in such a way that all the paths are separated from each other and each symbol experiences all paths coefficient. This separability is a unique feature of our scheme and cannot be achieved by other DAFT-based schemes. DAFT plays a fundamental role in AFDM, similarly to FT in OFDM. This work aims at establishing that AFDM is a promising new waveform for high mobility environments, having as well potential for communication at high frequency bands [23]. The contributions of this paper are summarized as follows:

- Introducing the Affine Fourier transform, we show how its discrete version can be achieved. Then, for the proposed AFDM, we analyze the DAFT domain input-output relation under doubly dispersive channels. The input-output relation is instrumental in giving insight on how DAFT parameters need to be tuned to avoid that time-domain channel paths with distinct delays or Doppler shifts overlap in the DAFT domain.
- We derive the diversity order of AFDM under maximum likelihood (ML) detection and we analytically show that AFDM achieves the optimal diversity order of the LTV channels.
- We propose a low complexity detection algorithm for AFDM taking advantage of its inherent channel sparsity. For that, the channel matrix is approximated as a band matrix placing some null symbols - zero-padding the AFDM frame - in the DAFT domain. We propose a low complexity iterative decision feedback equalizer (DFE) based on weighted maximal ratio combining (MRC) of the channel impaired input symbols received from different paths. The overall complexity of this algorithm is linear both in the number of subcarriers and in the number of paths. We also show that this detector has similar performance as LMMSE detector with much less complexity.
- For the embedded channel estimation, we arrange one pilot symbol and data symbols in one AFDM frame considering zero-padded symbols as guard intervals separating the pilot symbol and data symbols to avoid interference between them. We propose efficient approximated ML algorithms for channel estimation for the integer and fractional Doppler shifts. The proposed channel estimation schemes result in marginal performance degradation compared to AFDM with perfect channel knowledge.

This paper is organized as follows. In Section II, AFT and DAFT are introduced and the proposed AFDM is presented in Section III. Diversity analysis of AFDM in LTV channels is provided in Section IV. The proposed low complexity detection and channel estimation methods are presented in Section V and Section VI, respectively. In Section VII, simulation results for the AFDM performance are provided, and Section VIII concludes this paper.

---

## II. Affine Fourier Transform

In this section, we introduce the AFT and the DAFT, which form the basis of AFDM.

### A. Continuous Affine Fourier Transform

Affine Fourier Transform, also known as Linear Canonical Transform (LCT) [24], is a four-parameter $(a, b, c, d)$ class of linear integral transform defined as

$$
S_{a,b,c,d}(u) = \begin{cases} \displaystyle\int_{-\infty}^{+\infty} s(t) K_{a,b,c,d}(t, u) \, dt, & b \neq 0 \\[6pt] \displaystyle\frac{s(du) \, e^{-\imath \frac{cd}{2} u^2}}{\sqrt{a}}, & b = 0 \end{cases} \tag{1}
$$

where $(a, b, c, d)$ forms $\mathbf{M} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ with unit determinant, i.e., $ad - bc = 1$ and transform kernel given by

$$
K_{a,b,c,d}(t, u) = \frac{1}{\sqrt{2\pi|b|}} \, e^{-\imath\left(\frac{a}{2b} u^2 + \frac{1}{b} ut + \frac{d}{2b} t^2\right)}. \tag{2}
$$

The inverse transform can be expressed as an AFT having the parameters $\mathbf{M}^{-1} = \begin{pmatrix} a & -b \\ -c & d \end{pmatrix}$

$$
s(t) = \int_{-\infty}^{+\infty} S_{a,b,c,d}(u) K^*_{a,b,c,d}(t, u) \, du. \tag{3}
$$

The AFT generalizes several known mathematical transforms, such as Fourier transform $(0, 1/2\pi, -2\pi, 0)$, Laplace transform $(0, j(1/2\pi), j2\pi, 0)$, $\theta$-order fractional Fourier transform $(\cos\theta, (1/2\pi)\sin\theta, -2\pi\sin\theta, \cos\theta)$, Fresnel transform and the scaling operations. The extra degree of freedom of AFT provides flexibility and has been employed in many applications, including filter design, time-frequency analysis, phase retrievals, and multiplexing in communication. The effect of AFT can be interpreted by the Wigner distribution function (WDF). If $W_s(u, v)$ and $W_{S_{a,b,c,d}}(u, v)$ are the WDF of $s(t)$ and $S_{a,b,c,d}(u)$, respectively, then $W_{S_{a,b,c,d}}(u, v) = W_s(du - bv, -cu + av)$ where

$$
W_s(u, v) \overset{\text{def.}}{=} \frac{1}{2\pi} \int_{-\infty}^{+\infty} s(u + \tau/2) s^*(u - \tau/2) e^{-\imath v \tau} \, d\tau. \tag{4}
$$

Another way to express this is to say that the physical meaning of the AFT is to twist the time-frequency distribution of a function. After performing the AFT, the WDF of a function is twisted, but the area is unchanged.

### B. Discrete Affine Fourier Transform

The discrete transform can generally be used either to compute the continuous transform for spectral analysis or to process discrete data signals. Sampling the continuous function provides the input of the discrete transform in the former case, while a pure discrete sequence is considered for the input in the latter case. Therefore, discrete AFT is obtained in two types [25], which are essentially identical with different parameterizations. To derive the DAFT, input function $s(t)$ and $S_{a,b,c,d}(u)$ are sampled by the interval $\Delta t$ and $\Delta u$ as

$$
s_n = s(n\Delta t), \quad S_m = S_{a,b,c,d}(m\Delta u), \tag{5}
$$

where $n = 0, \ldots, N-1$ and $m = 0, \ldots, M-1$. From (5), we can convert (1) as

$$
S_m = \frac{1}{\sqrt{2\pi|b|}} \cdot \Delta t \cdot e^{-\imath\left(\frac{a}{2b} m^2 \Delta u^2\right)} \sum_{n=0}^{N-1} e^{-\imath\left(\frac{1}{b} mn \Delta u \Delta t + \frac{d}{2b} n^2 \Delta t^2\right)} s_n. \tag{6}
$$

This equation can be written in the form of transformation matrix

$$
S_m = \sum_{n=0}^{N-1} F_{a,b,c,d}(m, n) s_n, \tag{7}
$$

where $F_{a,b,c,d}(m, n) = \frac{1}{\sqrt{2\pi|b|}} \cdot \Delta t \cdot e^{-\imath\left(\frac{a}{2b} m^2 \Delta u^2 + \frac{1}{b} mn \Delta u \Delta t + \frac{d}{2b} n^2 \Delta t^2\right)}$.

In order for (7) to be reversible, the following condition should hold [25]

$$
\Delta t \, \Delta u = \frac{2\pi|b|}{M}. \tag{8}
$$

Thus, the DAFT of the first type can be written as follows:

$$
S_m = \frac{1}{\sqrt{M}} e^{-\imath \frac{a}{2b} m^2 \Delta u^2} \sum_{n=0}^{N-1} e^{-\imath\left(\frac{2\pi}{M} mn + \frac{d}{2b} n^2 \Delta t^2\right)} s_n, \quad b > 0 \tag{9}
$$

$$
S_m = \frac{1}{\sqrt{M}} e^{-\imath \frac{a}{2b} m^2 \Delta u^2} \sum_{n=0}^{N-1} e^{-\imath\left(-\frac{2\pi}{M} mn + \frac{d}{2b} n^2 \Delta t^2\right)} s_n, \quad b < 0. \tag{10}
$$

The DAFT of the second type [25] can be obtained by defining $c_1 = \frac{d}{4\pi b} \Delta t^2$ and $c_2 = \frac{a}{4\pi b} \Delta u^2$ so that $S_m$ in (7) writes as $S_m = \sum_{n=0}^{N-1} F_{c_1, c_2}(n, m) s_n$, where

$$
F_{c_1, c_2}(n, m) \triangleq \frac{1}{\sqrt{M}} e^{-\imath 2\pi\left(c_2 m^2 + \frac{\text{sgn}(b)}{M} mn + c_1 n^2\right)}. \tag{11}
$$

The condition in (8) then becomes $c_1 c_2 = \frac{ad}{4M^2}$. Since $a$ and $d$ can take any real value as long as $b$ and $c$ are adjusted to satisfy $ad - bc = 1$, there is no constraint for $c_1$ and $c_2$ and they can take any real values. Further simplification follows from fixing $\text{sgn}(b) = 1$, i.e., the DAFT is defined as

$$
S_m = \frac{1}{\sqrt{M}} e^{-\imath 2\pi c_2 m^2} \sum_{n=0}^{N-1} e^{-\imath 2\pi\left(\frac{1}{M} mn + c_1 n^2\right)} s_n, \tag{12}
$$

where $M \geq N$ and its inverse transform is the following

$$
s_n = \frac{1}{\sqrt{M}} e^{\imath 2\pi c_1 n^2} \sum_{m=0}^{M-1} e^{\imath 2\pi\left(\frac{1}{M} mn + c_2 m^2\right)} S_m. \tag{13}
$$

Moreover, we should take into account that sampling in one domain imposes periodicity in another domain. Considering (12) and (13), the following periodicity can be seen

$$
S_{m+kM} = e^{-\imath 2\pi c_2 (k^2 M^2 + 2kMm)} S_m, \tag{14}
$$

$$
s_{n+kN} = e^{\imath 2\pi c_1 (k^2 N^2 + 2kNn)} s_n. \tag{15}
$$

For our purposes, only constraint (15) matters, whose sole practical effect is on the kind of prefix one should add to a DAFT-based multicarrier symbol. When $M = N$, as considered in this paper, the inverse transform is the same as the forward transform with parameters $-c_1$ and $-c_2$ and conjugating the Fourier transform term. In matrix representation, arranging samples $s_n$ and $S_m$ in the period $[0, N)$ in vectors

$$
\mathbf{s} = (s_0, s_1, \ldots, s_{N-1}), \tag{16}
$$

$$
\mathbf{S} = (S_0, S_1, \ldots, S_{N-1}), \tag{17}
$$

DAFT is expressed as $\mathbf{S} = \mathbf{A}\mathbf{s}$ with $\mathbf{A} = \boldsymbol{\Lambda}_{c_2} \mathbf{F} \boldsymbol{\Lambda}_{c_1}$, $\mathbf{F}$ being the DFT matrix with entries $e^{-\imath 2\pi mn/N}/\sqrt{N}$ and

$$
\boldsymbol{\Lambda}_c = \text{diag}(e^{-\imath 2\pi c n^2}, \; n = 0, 1, \ldots, N-1). \tag{18}
$$

The inverse of the matrix $\mathbf{A}$ is given by $\mathbf{A}^{-1} = \mathbf{A}^H = \boldsymbol{\Lambda}_{c_1}^H \mathbf{F}^H \boldsymbol{\Lambda}_{c_2}^H$. We can now show that $\{F_{c_1,c_2}(n, m)\}_{m=0\cdots M-1}$ with $\text{sgn}(b) = 1$ and $M = N$ forms an orthonormal basis of $\mathbb{C}^N$, i.e.,

$$
\sum_{n=0}^{N-1} F_{c_1,c_2}(n, m_1) F^*_{c_1,c_2}(n, m_2) = \frac{1}{N} e^{-\imath 2\pi c_2 (m_1^2 - m_2^2)} \times \sum_{n=0}^{N-1} e^{-\imath \frac{2\pi}{N}(m_1 - m_2)n} = \delta(m_1 - m_2). \tag{19}
$$

With the setting $M = N$, the DAFT can thus be used to define a multi-carrier signal which i) has inter-carrier orthogonality and ii) can be generated (respectively received) with a transmitter (respectively receiver) whose main building block is an IDFT (respectively DFT) module.

---

## III. Affine Frequency Division Multiplexing

In this section, we present our DAFT-based multicarrier waveform and transceiver, coined AFDM. In this scheme, inverse DAFT (IDAFT) is used to map data symbols into the time domain, while DAFT is performed at the receiver to obtain the effective discrete affine Fourier domain channel response to the transmitted data, as shown in Fig. 1.

> **Fig. 1.** AFDM block diagram.

### A. Modulation

Let $\mathbf{x} \in \mathcal{A}^{N \times 1}$ denote the vector of information symbols in the discrete affine Fourier domain, where $\mathcal{A} \subset \mathbb{Z}[j]$ represents the alphabet and $\mathbb{Z}[j]$ denotes the number field whose elements have the form $z_r + z_i j$, with $z_r$ and $z_i$ integers. QAM symbols are considered in the remainder. The modulated signal can be written as

$$
s[n] = \sum_{m=0}^{N-1} x[m] \phi_n(m), \quad n = 0, \cdots, N-1, \tag{20}
$$

where $\phi_n(m) = \frac{1}{\sqrt{N}} \cdot e^{\imath 2\pi(c_1 n^2 + c_2 m^2 + nm/N)}$. In matrix form, (20) becomes $\mathbf{s} = \mathbf{A}^H \mathbf{x} = \boldsymbol{\Lambda}_{c_1}^H \mathbf{F}^H \boldsymbol{\Lambda}_{c_2}^H \mathbf{x}$.

Similarly to OFDM, the proposed scheme requires a prefix to combat multipath propagation and make the channel seemingly lie in a periodic domain. Due to different signal periodicity, a chirp-periodic prefix (CPP) is used here instead of an OFDM cyclic prefix (CP). For that, an $L_{cp}$-long prefix, occupying the positions of the negative-index time-domain samples, should be transmitted, where $L_{cp}$ is any integer greater than or equal to the value in samples of the maximum delay spread of the channel. With the periodicity defined in (15), the prefix is

$$
s[n] = s[N + n] e^{-\imath 2\pi c_1(N^2 + 2Nn)}, \quad n = -L_{cp}, \cdots, -1. \tag{21}
$$

Note that a CPP is simply a CP whenever $2Nc_1$ is an integer value and $N$ is even.

### B. Channel

After parallel to serial conversion and transmission over the channel, the received samples are

$$
r[n] = \sum_{l=0}^{\infty} s[n - l] g_n(l) + w[n], \tag{22}
$$

where $w_n \sim \mathcal{CN}(0, N_0)$ is an additive Gaussian noise and

$$
g_n(l) = \sum_{i=1}^{P} h_i e^{-\imath 2\pi f_i n} \delta(l - l_i), \tag{23}
$$

is the impulse response of channel at time $n$ and delay $l$, where $P \geq 1$ is the number of paths, $\delta(\cdot)$ is the Dirac delta function, and $h_i$, $f_i$ and $l_i$ are the complex gain, Doppler shift (in digital frequencies), and the integer delay associated with the $i$-th path, respectively. Note that this model is general and also covers the case where each delay tap can have a Doppler frequency spread by simply allowing for different paths $i, j \in \{1, \ldots, P\}$ to have the same delay $l_i = l_j$, while satisfying $f_i \neq f_j$. We define $\nu_i \triangleq N f_i = \alpha_i + a_i$, where $\nu_i \in [-\nu_{\max}, \nu_{\max}]$ is the Doppler shift normalized with respect to the subcarrier spacing, $\alpha_i \in [-\alpha_{\max}, \alpha_{\max}]$ is its integer part whereas $a_i$ is the fractional part satisfying $-\frac{1}{2} < a_i \leq \frac{1}{2}$. We assume that the maximum delay of the channel satisfies $l_{\max} \triangleq \max(l_i) < N$, and that the CPP length is greater than $l_{\max} - 1$.

After discarding the CPP, we can write (22) in the matrix form

$$
\mathbf{r} = \mathbf{H}\mathbf{s} + \mathbf{w}, \tag{24}
$$

with $\mathbf{w} \sim \mathcal{CN}(0, N_0 \mathbf{I})$, $\mathbf{H} = \sum_{i=1}^{P} h_i \boldsymbol{\Gamma}_{\text{CPP}_i} \boldsymbol{\Delta}_{f_i} \boldsymbol{\Pi}^{l_i}$, and $\boldsymbol{\Pi}$ is the forward cyclic-shift matrix

$$
\boldsymbol{\Pi} = \begin{pmatrix} 0 & \cdots & 0 & 1 \\ 1 & \cdots & 0 & 0 \\ \vdots & \ddots & \vdots & \vdots \\ 0 & \cdots & 1 & 0 \end{pmatrix}_{N \times N}, \tag{25}
$$

$\boldsymbol{\Delta}_{f_i} \triangleq \text{diag}(e^{-\imath 2\pi f_i n}, \; n = 0, 1, \ldots, N-1)$ and $\boldsymbol{\Gamma}_{\text{CPP}_i}$ is a $N \times N$ diagonal matrix

$$
\boldsymbol{\Gamma}_{\text{CPP}_i} = \text{diag}\left(\begin{cases} e^{-\imath 2\pi c_1(N^2 - 2N(l_i - n))} & n < l_i \\ 1 & n \geq l_i \end{cases}, \; n = 0, \ldots, N-1\right). \tag{26}
$$

From (26) we can see that whenever $2Nc_1$ is an integer and $N$ is even, $\boldsymbol{\Gamma}_{\text{CPP}_i} = \mathbf{I}$.

### C. Demodulation

At the receiver side, the DAFT domain output symbols are obtained by

$$
y[m] = \sum_{n=0}^{N-1} r[n] \phi_n^*(m). \tag{27}
$$

In matrix representation, the output can be written as

$$
\mathbf{y} = \mathbf{A}\mathbf{r} = \sum_{i=1}^{P} h_i \mathbf{A} \boldsymbol{\Gamma}_{\text{CPP}_i} \boldsymbol{\Delta}_{f_i} \boldsymbol{\Pi}^{l_i} \mathbf{A}^H \mathbf{x} + \mathbf{A}\mathbf{w} = \mathbf{H}_{\text{eff}} \mathbf{x} + \widetilde{\mathbf{w}}, \tag{28}
$$

where $\mathbf{H}_{\text{eff}} \triangleq \mathbf{A}\mathbf{H}\mathbf{A}^H$ and $\widetilde{\mathbf{w}} = \mathbf{A}\mathbf{w}$. Since $\mathbf{A}$ is a unitary matrix, $\widetilde{\mathbf{w}}$ and $\mathbf{w}$ have the same statistical properties.

### D. Input-Output Relation

From (28), we see that the received symbols are a linear combination of the transmitted symbols. Moreover, we know that features, such as diversity order, detection complexity, and channel estimation, are determined by the input-output relation, i.e., the structure of the effective channel. For example, the OFDM effective channel is diagonal, exhibiting poor diversity while the detection can be implemented using a 1-tap equalizer. For that, we provide here the structure of $\mathbf{H}_{\text{eff}}$ as input-output relation and show that it has a sparse structure and can be formed by the AFDM parameters. Considering the definition of $\mathbf{H}_{\text{eff}}$, (28) can be rewritten as

$$
\mathbf{y} = \sum_{i=1}^{P} h_i \mathbf{H}_i \mathbf{x} + \widetilde{\mathbf{w}}, \tag{29}
$$

where $\mathbf{H}_i \triangleq \mathbf{A} \boldsymbol{\Gamma}_{\text{CPP}_i} \boldsymbol{\Delta}_{f_i} \boldsymbol{\Pi}^{l_i} \mathbf{A}^H$. It can be shown that $H_i[p, q]$ is given by

$$
H_i[p, q] = \frac{1}{N} e^{\imath \frac{2\pi}{N}(Nc_1 l_i^2 - q l_i + Nc_2(q^2 - p^2))} \mathcal{F}_i(p, q), \tag{30}
$$

where we denote $\mathcal{F}_i(p, q)$ as

$$
\mathcal{F}_i(p, q) = \sum_{n=0}^{N-1} e^{-\imath \frac{2\pi}{N}((p - q + \nu_i + 2Nc_1 l_i)n)} = \frac{e^{-\imath 2\pi(p - q + \nu_i + 2Nc_1 l_i)} - 1}{e^{-\imath \frac{2\pi}{N}(p - q + \nu_i + 2Nc_1 l_i)} - 1}. \tag{31}
$$

As we can see, the value of $\mathcal{F}_i(p, q)$ depends on the Doppler shift $\nu_i$. Therefore, we have two cases, namely integer Doppler shift and fractional Doppler shift. We first show the input-output relation for the integer case, and we state the relation of the general case afterwards.

#### 1) Integer Doppler Shifts

With $\nu_i$ being integer valued for all $i \in \{1, \ldots, P\}$, i.e., $a_i = 0$, (31) is equal to

$$
\mathcal{F}_i(p, q) = \begin{cases} N & q = (p + \text{loc}_i)_N \\ 0 & \text{otherwise} \end{cases} \tag{32}
$$

where $\text{loc}_i \triangleq (\alpha_i + 2Nc_1 l_i)_N$, $(\cdot)_N$ is the modulo $N$ operation and (30) writes as

$$
H_i[p, q] = \begin{cases} e^{\imath \frac{2\pi}{N}(Nc_1 l_i^2 - q l_i + Nc_2(q^2 - p^2))} & q = (p + \text{loc}_i)_N \\ 0 & \text{otherwise.} \end{cases} \tag{33}
$$

Hence, there is only one non-zero element in each row of $\mathbf{H}_i$ as shown in Fig. 2a, and the input-output relation for (29) becomes

$$
y[p] = \sum_{i=1}^{P} h_i e^{\imath \frac{2\pi}{N}(Nc_1 l_i^2 - q l_i + Nc_2(q^2 - p^2))} x[q] + \widetilde{w}[p], \quad 0 \leq p \leq N-1, \tag{34}
$$

where $q = (p + \text{loc}_i)_N$.

#### 2) Fractional Doppler Shifts

Considering the fractional Doppler shifts, it can be shown that for a given $p$, $\mathcal{F}_i(p, q) \neq 0$, for all $q$. However, the magnitude of $H_i[p, q]$ has a peak at $q = (p + \text{loc}_i)_N$ and decreases as $q$ moves away from $\text{loc}_i$. To show this, we have

$$
|H_i[p, q]| = \left|\frac{e^{\imath \frac{2\pi}{N}(Nc_1 l_i^2 - q l_i + Nc_2(q^2 - p^2))}}{N} \mathcal{F}_i(p, q)\right| = \left|\frac{1}{N} \mathcal{F}_i(p, q)\right| = \left|\frac{\sin(N\theta)}{N\sin(\theta)}\right|, \tag{35}
$$

where $\theta \triangleq \frac{\pi}{N}(p - q + \text{loc}_i + a_i)$. Using the inequality $|\sin(N\theta)| \leq |N\sin(\theta)|$, which is tight for small values of $\theta$, we can show

$$
\left|\frac{\sin(N\theta)}{N\sin(\theta)}\right| = \left|\frac{\sin((N-1)\theta)\cos(\theta) + \sin(\theta)\cos((N-1)\theta)}{N\sin(\theta)}\right| \leq \frac{N-1}{N}|\cos(\theta)| + \frac{1}{N}. \tag{36}
$$

The right-hand side (r.h.s.) of (36) has its peak at the smallest $|\theta_{p,q,i}|$ when $q = (p + \text{loc}_i)_N$. As $q$ moves away from $(p + \text{loc}_i)_N$, $|\theta_{p,q,i}|$ increases and the r.h.s. of (36) decreases. Moreover, the larger the value of $N$, the faster this decrease is with respect to $q$. Therefore, we consider from now on that $|H_i(p, q)|$ is non-zero only for $2k_\nu + 1$ values of $q$ corresponding to an interval centered at $q = (p + \text{loc}_i)_N$. Here, $k_\nu$ is chosen as a function of $N$ in such a way that for all $i$ and $p$ the r.h.s. of (36) is smaller than a sensitivity threshold for all $|q - (p + \text{loc}_i)_N| > k_\nu$. In formulas, this translates to the matrix written in (37):

$$
H_i[p, q] = \begin{cases} \displaystyle\frac{e^{\imath \frac{2\pi}{N}(Nc_1 l_i^2 - q l_i + Nc_2(q^2 - p^2))}}{N} \mathcal{F}_i(p, q) & (p + \text{loc}_i - k_\nu)_N \leq q \leq (p + \text{loc}_i + k_\nu)_N \\[6pt] 0 & \text{otherwise} \end{cases} \tag{37}
$$

Hence, there are $2k_\nu + 1$ non-zero elements in each row of $\mathbf{H}_i$, as shown in Fig. 2b, and the input-output relation for (29) is written as

$$
y[p] = \sum_{i=1}^{P} \frac{1}{N} h_i e^{\imath 2\pi c_1 l_i^2} \sum_{q=(p+\text{loc}_i - k_\nu)_N}^{(p+\text{loc}_i + k_\nu)_N} e^{\imath \frac{2\pi}{N}(-q l_i + Nc_2(q^2 - p^2))} \times \frac{e^{\imath 2\pi(p - q + a_i + \text{loc}_i)} - 1}{e^{\imath \frac{2\pi}{N}(p - q + a_i + \text{loc}_i)} - 1} x[q] + \widetilde{w}[p], \quad 0 \leq p \leq N-1. \tag{38}
$$

It should be noted that the range for the sum is circulant, i.e., when it is from $N-3$ to $1$, it is counted as $N-3, N-2, N-1, 0, 1$.

### E. AFDM Parameters

The performance of DAFT-based modulation schemes critically depends on the choice of parameters $c_1$ and $c_2$. For instance, OCDM uses $c_1 = c_2 = \frac{1}{2N}$, whereas in DAFT-OFDM $c_2 = 0$ and $c_1$ are adapted to the delay-Doppler channel profile to minimize ICI. Nevertheless, both schemes fail to achieve the optimal diversity order in LTV channels, as shown in Section VII. In the proposed AFDM, we find $c_1$ and $c_2$ for which the DAFT domain impulse response constitutes a full delay-Doppler representation of the channel. This allows AFDM to achieve the optimal diversity order in LTV channels, as shown in Section IV. For that, the non-zero entries in each row of $\mathbf{H}_i$ for each path $i \in \{1, \ldots, P\}$ should not coincide with the position of the non-zero entries of the same row of $\mathbf{H}_j$ for any $j \in \{1, \ldots, P\}$ such that $j \neq i$. Observing (33) and (37), the location of each path depends on its delay-Doppler information and AFDM parameters. For the integer and fractional Doppler shifts, $\text{loc}_i$ and $\text{loc}_{i,frac}$ are in the following range

$$
-\alpha_{\max} + 2Nc_1 l_i \leq \text{loc}_i \leq \alpha_{\max} + 2Nc_1 l_i, \tag{39}
$$

$$
-\alpha_{\max} - k_\nu + 2Nc_1 l_i \leq \text{loc}_{i,frac} \leq \alpha_{\max} + k_\nu + 2Nc_1 l_i, \tag{40}
$$

respectively. Therefore, for the positions of the non-zero entries of $\mathbf{H}_i$ and $\mathbf{H}_j$ to not overlap, the intersection of the corresponding ranges of $\text{loc}_i$ and $\text{loc}_j$ for the integer case and $\text{loc}_{i,frac}$ and $\text{loc}_{j,frac}$ for the fractional case should be empty, i.e.,

$$
\{-\alpha_{\max} + 2Nc_1 l_i, \ldots, \alpha_{\max} + 2Nc_1 l_i\} \cap \{-\alpha_{\max} + 2Nc_1 l_j, \ldots, \alpha_{\max} + 2Nc_1 l_j\} = \emptyset, \tag{41}
$$

$$
\{-\alpha_{\max} - k_\nu + 2Nc_1 l_i, \ldots, \alpha_{\max} + k_\nu + 2Nc_1 l_i\} \cap \{-\alpha_{\max} - k_\nu + 2Nc_1 l_j, \ldots, \alpha_{\max} + k_\nu + 2Nc_1 l_j\} = \emptyset. \tag{42}
$$

For the paths with different delays, assuming $l_i < l_j$, satisfying (41) and (42) is equivalent to satisfying the constraints

$$
2Nc_1 > \frac{2\alpha_{\max}}{l_j - l_i}, \tag{43}
$$

$$
2Nc_1 > \frac{2(\alpha_{\max} + k_\nu)}{l_j - l_i}. \tag{44}
$$

If there is sparsity in the time-domain impulse response of the channel, for the integer case $c_1$ is set to

$$
c_1 = \frac{2\alpha_{\max} + 1}{2N \min_{i,j}(l_j - l_i)}, \tag{45}
$$

and for the fractional case $c_1$ is chosen as

$$
c_1 = \frac{2(\alpha_{\max} + \xi_\nu) + 1}{2N \min_{i,j}(l_j - l_i)}, \tag{46}
$$

for some $\xi_\nu \leq k_\nu$ whose value is discussed later on. In the case of no delay sparsity, then the minimum value of $l_j - l_i$ is one, then (45) and (46) simplify to

$$
c_1 = \frac{2\alpha_{\max} + 1}{2N}, \tag{47}
$$

$$
c_1 = \frac{2(\alpha_{\max} + \xi_\nu) + 1}{2N}, \tag{48}
$$

respectively. Through $\xi_\nu$, there is flexibility in setting $c_1$ and reducing pilot overhead (see Section VI) at the expense of $|\mathbf{H}_{\text{eff}}|$ no longer being strictly circulant. Moreover, the only remaining condition for the DAFT-domain impulse response to constitute a full delay-Doppler representation of the channel is to ensure that the non-zero entries of any two matrices $\mathbf{H}_{i_{\min}}$ and $\mathbf{H}_{i_{\max}}$ corresponding to paths $i_{\min}$ and $i_{\max}$ with delays $l_{i_{\min}} \triangleq \min_{i=1\cdots P} l_i$ and $l_{i_{\max}} \triangleq \max_{i=1\cdots P} l_i$, respectively, do not overlap due to the modular operation in (33) and (37). This overlapping never occurs if $2\alpha_{\max} l_{\max} + 2\alpha_{\max} + l_{\max} < N$ for the integer case and $2(\alpha_{\max} + k_\nu) l_{\max} + 2(\alpha_{\max} + k_\nu) + l_{\max} < N$ for the fractional case. Since wireless channels are usually underspread (i.e., $l_{\max} \ll N$ and $\alpha_{\max} \ll N$), this condition is satisfied even for moderate values of $N$.

With this parameter setting, channel paths with different delay values or different Doppler frequency shifts become separated in the DAFT domain, resulting in $\mathbf{H}_{\text{eff}}$ having the structure shown in Fig. 3 (for the fractional case, $\alpha_{\max}$ should be replaced with $\alpha_{\max} + \xi_\nu$). Thus, we get a delay-Doppler representation of the channel in the DAFT domain since the delay-Doppler profile can be determined from the positions of the non-zero entries in any row of $\mathbf{H}_{\text{eff}}$. This feature can be obtained by neither DAFT-OFDM (since by construction the effective channel matrix is made as close to diagonal as possible to reduce ICI), nor OCDM (since setting $c_1 = \frac{1}{2N}$, there may exist two paths $i \neq j$ such that the non-zero entries of $\mathbf{H}_i$ and $\mathbf{H}_j$ coincide under some delay-Doppler profiles of the channel). In the next section, we show that this unique feature of AFDM translates into being diversity order optimal in LTV channels. Also, this parameter setting results in the subcarriers $\phi_n(m) = \frac{1}{\sqrt{N}} \cdot e^{\imath 2\pi(c_1 n^2 + c_2 m^2 + nm/N)}$ having a time-frequency content that is distinct from all so-far existing waveforms. This time-frequency content is illustrated in Fig. 4 and is compared to that of OCDM and OFDM.

> **Fig. 2.** Structure of $\mathbf{H}_i$ for (a) integer and (b) fractional Doppler shifts.

> **Fig. 3.** Structure of $\mathbf{H}_{\text{eff}}$ in AFDM for the integer case.

> **Fig. 4.** Time-frequency representation of OFDM, OCDM, and AFDM subcarriers.

---

## IV. Diversity Analysis

In this section, we analyze the diversity order of AFDM. Qualitatively speaking, the diversity order of a waveform is the slope of its bit error rate (BER) performance curve in the high-SNR regime. Its more precise definition is given in (53) based on the pairwise error probability (PEP). Due to space limitations, diversity analysis is presented only in the case of integer Doppler shifts. However, Theorem 1 given below, also holds for the fractional Doppler case. To this end, we rewrite (28) as

$$
\mathbf{y} = \sum_{i=1}^{P} h_i \mathbf{H}_i \mathbf{x} + \widetilde{\mathbf{w}} = \boldsymbol{\Phi}(\mathbf{x})\mathbf{h} + \widetilde{\mathbf{w}}, \tag{49}
$$

where $\mathbf{h} = [h_1, h_2, \ldots, h_P]$ is a $P \times 1$ vector and $\boldsymbol{\Phi}(\mathbf{x})$ is the $N \times P$ concatenated matrix

$$
\boldsymbol{\Phi}(\mathbf{x}) = [\mathbf{H}_1 \mathbf{x} \;|\; \ldots \;|\; \mathbf{H}_P \mathbf{x}]. \tag{50}
$$

We now express PEP. First, we normalize the elements of $\mathbf{x}$ so that the average energy of $\mathbf{x}$ is one, thus the signal-to-noise ratio (SNR) is given by $\frac{1}{N_0}$. Assuming perfect channel state information and ML detection at the receiver, the conditional PEP between $\mathbf{x}_m$ and $\mathbf{x}_n$, i.e., transmitting symbol $\mathbf{x}_m$ and deciding in favor of $\mathbf{x}_n$ at the receiver, can be upper bounded as

$$
P(\mathbf{x}_m \to \mathbf{x}_n) \leq \prod_{l=1}^{r} \frac{1}{1 + \frac{\lambda_l^2}{4PN_0}} \tag{51}
$$

where $\lambda_l$ is the $l$-th singular value of the matrix $\boldsymbol{\Phi}(\boldsymbol{\delta}^{(m,n)})$ and $\boldsymbol{\delta}^{(m,n)} = \mathbf{x}_m - \mathbf{x}_n$. At high SNR, (51) becomes

$$
P(\mathbf{x}_m \to \mathbf{x}_n) \leq \frac{1}{\left(\frac{1}{N_0}\right)^r \prod_{l=1}^{r} \frac{\lambda_l^2}{4P}}. \tag{52}
$$

We can see from (52) that the exponent of the SNR term, $\frac{1}{N_0}$, is $r$, which is equal to the rank of the matrix $\boldsymbol{\Phi}(\boldsymbol{\delta}^{(m,n)})$. The overall BER is dominated by the PEP with the minimum value of $r$, for all $m, n, m \neq n$. Hence, the diversity order of AFDM is given by

$$
\rho = \min_{m,n \; m \neq n} \text{rank}(\boldsymbol{\Phi}(\boldsymbol{\delta}^{(m,n)})). \tag{53}
$$

First, we show that for DAFT-based multi-carrier schemes, a necessary (but not sufficient) condition to achieve the optimal diversity order, i.e., $\rho = P$, is

$$
\forall i, j \in [1, \cdots, P], \quad \text{loc}_i \neq \text{loc}_j. \tag{54}
$$

The above condition can hold for AFDM, but not for OCDM and DAFT-OFDM. For that sake, we assume that there exist channel paths $i$ and $j$ such that the locations of their corresponding non-zero elements in the channel matrix are the same, i.e., $\text{loc}_i = \text{loc}_j$. We then show that the optimal diversity order cannot be achieved under this assumption. Indeed, for the optimal diversity order, the matrix in (55), composed of the columns $i$ and $j$ of $\boldsymbol{\Phi}(\boldsymbol{\delta})$ should be of rank 2 for all possible values of $\boldsymbol{\delta}$ where $q_0 = \text{loc}_i = \text{loc}_j$.

$$
\begin{pmatrix} H_i[1, q_0]\delta[q_0] & H_j[1, q_0]\delta[q_0] \\ H_i[2, (q_0+1)_N]\delta[(q_0+1)_N] & H_j[2, (q_0+1)_N]\delta[(q_0+1)_N] \\ \vdots & \vdots \\ H_i[N, (q_0+N-1)_N]\delta[(q_0+N-1)_N] & H_j[N, (q_0+N-1)_N]\delta[(q_0+N-1)_N] \end{pmatrix} \tag{55}
$$

However, when $\boldsymbol{\delta}$ is such that it has a single non-zero element, the two columns of the matrix in (55) are dependent, hence the rank of this matrix cannot be 2. Consequently, the rank of $\boldsymbol{\Phi}(\boldsymbol{\delta})$ cannot be $P$ and the condition in (54) is thus necessary for the optimal diversity order. Therefore, proving that AFDM achieves the optimal diversity order is equivalent to proving that tuning $c_2$ can make matrix $\boldsymbol{\Phi}(\boldsymbol{\delta})$ to be full rank.

**Theorem 1:** For a linear time-varying channel with a maximum delay $l_{\max}$ and maximum normalized Doppler shift $\alpha_{\max}$, AFDM with $c_1$ satisfying (47) achieves the optimal diversity order ($\rho = P$) if

$$
2\alpha_{\max} + l_{\max} + 2\alpha_{\max} l_{\max} < N. \tag{56}
$$

*Proof:* See Appendix A. ∎

---

## V. Low-Complexity Weighted MRC-Based DFE Detection

Although for showing the diversity order of AFDM ML detection is used, it is prohibitively complex to implement in real-world communication systems. For that, in this section, we propose a low-complexity detector. The first step is to place some null symbols that allow approximating the truncated part of $\mathbf{H}_{\text{eff}}$ as a band matrix. This also simplifies the input-output relation as the modular operation is no longer needed (see Fig. 5). Note that these symbols do not entail extra overhead as they can serve not only the proposed detection algorithms but also embedded pilot aided channel estimation. Due to the structure of $\mathbf{H}_{\text{eff}}$, the number of the null guard symbols should be greater than

$$
Q \triangleq (l_{\max} + 1)(2(\alpha_{\max} + \xi_\nu) + 1) - 1. \tag{57}
$$

Taking into account the zero padding, the vector of DAFT domain received samples writes as

$$
\underline{\mathbf{y}} = \underline{\mathbf{H}}_{\text{eff}} \underline{\mathbf{x}} + \underline{\widetilde{\mathbf{w}}}, \tag{58}
$$

where $\underline{\mathbf{x}}$ and $\underline{\mathbf{H}}_{\text{eff}}$ are the truncated parts of $\mathbf{x}$ and $\mathbf{H}_{\text{eff}}$, respectively (see Fig. 5). They can be expressed using the matrix $\mathbf{T} = [\mathbf{I}_N]_{Q-(\alpha_{\max}+\xi_\nu):N-(\alpha_{\max}+\xi_\nu)-1,:}$ as $\underline{\mathbf{x}} = \mathbf{T}\mathbf{x}$ and $\underline{\mathbf{H}}_{\text{eff}} = \mathbf{H}_{\text{eff}}\mathbf{T}^H$.

Using LMMSE equalization for (58) requires $\mathcal{O}(N^3)$ flops, which can be prohibitive for large $N$. We hence propose a weighted MRC-based DFE exploiting the sparse representation of the communication channel provided by AFDM.

As shown in Fig. 5, $\underline{\mathbf{H}}_{\text{eff}}$ has $L$ non-zero entries per column, where $L = P$ for the integer Doppler shift case and $L = (2\xi_\nu + 1)P$ for fractional Doppler shift case, with $L \leq Q$ in both cases. Each of these non-zero entries of a column of $\underline{\mathbf{H}}_{\text{eff}}$ provides a copy of the data symbol corresponding to the index of this column. We propose a detection scheme where each data symbol is detected from a weighted MRC of its $L$ channel-impaired received copies. Fig. 6 depicts an instance of this detector for AFDM with $N = 8$ and a 3-path channel with $Q = 2$. The proposed detector is iterative, wherein each iteration, the estimated inter symbol interference is canceled in the branches selected for the combining. Considering the structure of $\underline{\mathbf{H}}_{\text{eff}}$, it can be seen that each received symbol $y[k]$ is given by

$$
y[k] = \sum_{i=0}^{L-1} \underline{H}_{\text{eff}}[k, p_k^i] x[p_k^i], \tag{59}
$$

where $p_k^i$ is the column index of the $i$-th path coefficient in row $k$ of matrix $\underline{\mathbf{H}}_{\text{eff}}$. Let $b_k^i$ be the channel impaired input symbol $x[k]$ in the received samples $y[q_k^i]$ after canceling the interference from other input symbols, where $q_k^i$ is the row index of the $i$-th path coefficient in column $k$ of matrix $\underline{\mathbf{H}}_{\text{eff}}$. In each iteration, assuming estimates of the input symbols $x[k]$ are available, either from the current iteration (for $p_{q_k^i}^j < k$, $j = 0, \ldots, L-1$) or previous iteration (for $p_{q_k^i}^j > k$, $j = 0, \ldots, L-1$), $b_k^i$ can be written as

$$
b_k^i = y[q_k^i] - \sum_{p_{q_k^i}^j < k} \underline{H}_{\text{eff}}[q_k^i, p_{q_k^i}^j] \hat{x}[p_{q_k^i}^j]^{(n)} - \sum_{p_{q_k^i}^j > k} \underline{H}_{\text{eff}}[q_k^i, p_{q_k^i}^j] \hat{x}[p_{q_k^i}^j]^{(n-1)}, \tag{60}
$$

where superscript $(n)$ denotes the $n$-th iteration. It can be seen that for each symbol $x[k]$, we need to compute $L$ scalars. This operation has complexity order of $\mathcal{O}(L^2)$. However, when computing $b_k^i$ for all symbols $k$, there are some redundant operations involved that can be avoided by instead computing $b_k^i$ as follows

$$
b_k^i = \Delta y[q_k^i]^{(n)} + \underline{H}_{\text{eff}}[q_k^i, k] \hat{x}[k]^{(n-1)}. \tag{61}
$$

Here, $\Delta y_{q_k(i)}^{(n)}$ is the residual error remaining while reconstructing the received symbols and is given by

$$
\Delta y[q_k^i]^{(n)} = y[q_k^i] - \sum_{p_{q_k^i}^j < k} \underline{H}_{\text{eff}}[q_k^i, p_{q_k^i}^j] \hat{x}[p_{q_k^i}^j]^{(n)} - \sum_{p_{q_k^i}^j \geq k} \underline{H}_{\text{eff}}[q_k^i, p_{q_k^i}^j] \hat{x}[p_{q_k^i}^j]^{(n-1)}. \tag{62}
$$

Define $g_k^{(n)}$ and $d$ as

$$
g_k^{(n)} \triangleq \sum_{i=0}^{L-1} \underline{H}_{\text{eff}}^*[q_k^i, k] b_k^i = \sum_{i=0}^{L-1} \underline{H}_{\text{eff}}^*[q_k^i, k] \Delta y[q_k^i]^{(n)} + d x[k]^{(n-1)}, \tag{63}
$$

$$
d \triangleq \sum_{i=0}^{L-1} |\underline{H}_{\text{eff}}[q_k^i, k]|^2. \tag{64}
$$

It should be noted that since $|\underline{\mathbf{H}}_{\text{eff}}|$ is a circulant matrix, the value of $d$ is independent of $k$ and needs to be computed only once. We now denote the SNR by $\gamma$. Instead of directly using $g_k^{(n)}/d$ as the estimate of $x_k$ (which would have amounted to using the MRC criterion), we define the symbol estimate as

$$
\hat{x}[k]^{(n)} = c_k^{(n)}, \tag{65}
$$

$$
c_k^{(n)} \triangleq \frac{g_k^{(n)}}{d + \gamma^{-1}} = \frac{1}{d + \gamma^{-1}} \sum_{i=0}^{L-1} \underline{H}_{\text{eff}}^*[q_k^i, k] \Delta y[q_k^i]^{(n)} + \frac{d}{d + \gamma^{-1}} \hat{x}[k]^{(n-1)}. \tag{66}
$$

We later show (see Section V-.2) that this weighting of $g_k^{(n)}$ while computing $\hat{x}_k^{(n)}$ guarantees that the iterative detection algorithm converges to the LMMSE estimate of the symbols vector $\mathbf{x}$. In each iteration, after estimation of each symbol $x[k]^{(n)}$, the values of $\Delta y[q_k^i]^{(n)}$ for $i = 0, \ldots, L-1$ need to be updated using

$$
\Delta y[q_k^i]^{(n)} = \Delta y[q_k^i]^{(n)} - \underline{H}_{\text{eff}}^*[q_k^i, k](x[k]^{(n)} - x[k]^{(n-1)}). \tag{67}
$$

Once all symbols are estimated, they are used for interference cancellation in the next iteration. The algorithm continues until the maximum number of iterations ($n_{\text{iter}}$) is reached or the updated input symbol vector is close enough (less than $\epsilon$) to the previous one, as summarized in Algorithm 1. Computing the complexity of Algorithm 1 is straightforward as it only involves scalar operations. Step 3 to step 8 requires $2L$ CMs, $3L + 1$ CAs and 1 CD. Therefore, its total complexity is $n_{\text{iter}}(5L + 1)(N - Q)$.

> **Fig. 5.** Truncated parts of $\mathbf{x}$ and $\underline{\mathbf{H}}_{\text{eff}}$.

> **Fig. 6.** Weighted MRC operation for $N = 8$ with a 3-path channel with $Q = 2$.

---

**Algorithm 1** Weighted MRC-Based DFE Detection

**Data:** $\underline{\mathbf{H}}_{\text{eff}}$, $d$, $\mathbf{y}$, $\hat{\mathbf{x}}^{(0)} = \mathbf{0}$, $\Delta\mathbf{y}^{(0)} = \mathbf{y}$

1. **for** $n = 1$: $n_{\text{iter}}$ **do**
2. &emsp; **for** $k = 0$: $N$-$Q$-$1$ **do**
3. &emsp;&emsp; $g_k^{(n)} = \sum_{i=0}^{L-1} \underline{H}_{\text{eff}}^*[q_k^i, k] \Delta y[q_k^i]^{(n)} + d x[k]^{(n-1)}$
4. &emsp;&emsp; $c_k^{(n)} = \frac{g_k^{(n)}}{d + \gamma^{-1}}$
5. &emsp;&emsp; $\hat{x}[k]^{(n)} = c_k^{(n)}$
6. &emsp;&emsp; **for** $i = 0$: $L$-$1$ **do**
7. &emsp;&emsp;&emsp; $\Delta y[q_k^i]^{(n)} = \Delta y[q_k^i]^{(n)} - \underline{H}_{\text{eff}}^*[q_k^i, k](x[k]^{(n)} - x[k]^{(n-1)})$
8. &emsp;&emsp; **end**
9. &emsp; **end**
10. &emsp; **if** $\|\hat{\mathbf{x}}^{(n)} - \hat{\mathbf{x}}^{(n-1)}\| < \epsilon$ **then** EXIT;
11. **end**

---

### 1) Convergence

The detector convergence is analyzed using properties of iterative methods for linear systems. To this end, Algorithm 1 can be expressed in the matrix form as

$$
\hat{\mathbf{x}}^{(n)} = \frac{d}{\gamma^{-1} + d} \hat{\mathbf{x}}^{(n-1)} + \frac{1}{\gamma^{-1} + d} (\underline{\mathbf{H}}_{\text{eff}}^H \mathbf{y} - \mathbf{L}\hat{\mathbf{x}}^{(n)} - (\mathbf{L}^H + \mathbf{D})\hat{\mathbf{x}}^{(n-1)}), \tag{68}
$$

where $\mathbf{D} = d\mathbf{I}$, $\mathbf{L}$ and $\mathbf{L}^H$ are the matrices containing diagonal elements, strictly lower and upper triangular parts of the Hermitian matrix $\underline{\mathbf{H}}_{\text{eff}}^H \underline{\mathbf{H}}_{\text{eff}}$, respectively. Equation (68) can be rewritten in the form

$$
\hat{\mathbf{x}}^{(n)} = -\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S})\hat{\mathbf{x}}^{(n-1)} + \mathbf{M}^{-1}\mathbf{b}, \tag{69}
$$

where $\mathbf{S} = (\gamma^{-1} + d)\mathbf{I} + \mathbf{L}$, $\mathbf{R} = \underline{\mathbf{H}}_{\text{eff}}^H \underline{\mathbf{H}}_{\text{eff}} + \gamma^{-1}\mathbf{I}$ and $\mathbf{b} = \underline{\mathbf{H}}_{\text{eff}}^H \mathbf{y}$. The iteration in (69) is convergent if the spectral radius of the matrix $-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S})$, denoted as $\rho(-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S}))$, is strictly smaller than one [26], [27].

**Theorem 2:** The iteration in (69) is convergent (i.e., $\rho(-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S})) < 1$) if $\mathbf{R} = \underline{\mathbf{H}}_{\text{eff}}^H \underline{\mathbf{H}}_{\text{eff}} + \gamma^{-1}\mathbf{I}$ is a positive definite Hermitian matrix.

*Proof:* See Appendix B. ∎

### 2) Relation to the Gauss-Seidel Method

LMMSE equalization is equivalent to solving the system of linear equations $(\underline{\mathbf{H}}_{\text{eff}}^H \underline{\mathbf{H}}_{\text{eff}} + \gamma^{-1}\mathbf{I})\mathbf{x} = \underline{\mathbf{H}}_{\text{eff}}^H \mathbf{y}$. One way to solve it is using the properties of the Gauss-Seidel iterative method for solving linear equations [26]. According to this method, decomposing $\underline{\mathbf{H}}_{\text{eff}}^H \underline{\mathbf{H}}_{\text{eff}} + \gamma^{-1}\mathbf{I}$ additively in its diagonal part $(d + \gamma^{-1}\mathbf{I})$, its strict lower triangular part $\mathbf{L}$ as well as its strict upper triangular part $\mathbf{L}^H$, gives $\mathbf{x}^{(n)}$ as

$$
\hat{\mathbf{x}}^{(n)} = -((\gamma^{-1} + d)\mathbf{I} + \mathbf{L})^{-1} \mathbf{L}^H \mathbf{x}^{(n-1)} + ((\gamma^{-1} + d)\mathbf{I} + \mathbf{L})^{-1} \underline{\mathbf{H}}_{\text{eff}}^H \mathbf{y}. \tag{70}
$$

This means that the weighted MRC-based DFE converges to the LMMSE estimate. This is confirmed in the simulation results section.

---

## VI. Embedded Channel Estimation

In order to perform detection, the channel matrix $\mathbf{H}_{\text{eff}}$ should be known at the receiver side. To enable that, we propose a channel estimation scheme based on the transmission of an embedded pilot symbol $x_{\text{pilot}}$ in each AFDM frame surrounded by $Q$ null guard samples on each side of $x_{\text{pilot}}$ (where $Q$ is defined in (57)) and $N - 1 - 2Q$ data symbols $x_0^{\text{data}}, \ldots, x_{N-2-Q}^{\text{data}}$. The guard samples separate the data symbols from the pilot symbol so that the channel estimation can be done at the receiver without any interference from the data symbols. Equivalently, data detection using the estimated channel is performed without interference from the pilot symbol.

We place $x_{\text{pilot}}$ as the first symbol in the frame as shown in Fig. 7:

$$
x[p] = \begin{cases} x_{\text{pilot}}, & p = 0 \\ 0, & 1 \leq p \leq Q, \; N_Q + 1 \leq p \leq N-1 \\ x_{p-Q-1}^{\text{data}}, & Q + 1 \leq p \leq N_Q, \end{cases} \tag{71}
$$

where $N_Q \triangleq N - Q - 1$. Considering the channel model defined in (23), three parameters of each path, delay, Doppler shift, and complex gain, i.e., $3P$ unknown parameters $\boldsymbol{\theta} = [h_0, \ldots, h_{P-1}, l_0, \ldots, l_{P-1}, \nu_0, \ldots, \nu_{P-1}]$ should be estimated. As shown in Fig. 8, the part of the received signal that is related to the pilot symbol is considered for the channel estimation. These symbols are expressed by the following equation

$$
\underline{\mathbf{y}}_E = \underline{\mathbf{H}}_{\text{eff},E} \underline{\mathbf{x}}_E + \underline{\widetilde{\mathbf{w}}}_E, \tag{72}
$$

where $E$ stands for "Estimation" phase, $\underline{\mathbf{x}}_E$, $\underline{\mathbf{y}}_E$ and $\underline{\mathbf{H}}_{\text{eff},E}$ are the parts of $\mathbf{x}$, $\mathbf{y}$ and $\mathbf{H}_{\text{eff}}$ related to the channel estimation, respectively. They can be expressed by the matrix $\mathbf{T}_{t,E} = [\mathbf{I}_N]_{\text{ind}_{t,E},:}$ and $\mathbf{T}_{r,E} = [\mathbf{I}_N]_{\text{ind}_{r,E},:}$ where $\text{ind}_{t,E} = [0 : Q \;\; N_Q + 1 : N-1]$ and $\text{ind}_{r,E} = [0 : \alpha_{\max} + \xi_\nu \;\; N_Q + \alpha_{\max} + \xi_\nu + 1 : N-1]$ as $\underline{\mathbf{x}}_E = \mathbf{T}_{t,E}\mathbf{x}$, $\underline{\mathbf{y}}_E = \mathbf{T}_{r,E}\mathbf{y}$ and $\underline{\mathbf{H}}_{\text{eff},E} = \mathbf{T}_{r,E}\mathbf{H}_{\text{eff}}\mathbf{T}_{t,E}^H$. Considering the ML detector, the log-likelihood function to be minimized is given by

$$
l(\underline{\mathbf{y}}_E | \boldsymbol{\theta}, \underline{\mathbf{x}}_E) = \|\underline{\mathbf{y}}_E - \underline{\mathbf{H}}_{\text{eff},E} \underline{\mathbf{x}}_E\|^2. \tag{73}
$$

Considering (49), it can be written as

$$
l(\underline{\mathbf{y}}_E | \boldsymbol{\theta}, \underline{\mathbf{x}}_E) = \left\|\underline{\mathbf{y}}_E - \sum_{i=0}^{P-1} h_i \underline{\mathbf{H}}_{i,E} \underline{\mathbf{x}}_E\right\|^2. \tag{74}
$$

where $\underline{\mathbf{H}}_{i,E} = \mathbf{T}_{r,E}\mathbf{H}_i\mathbf{T}_{t,E}^H$. Since $\underline{\mathbf{x}}_E$ has only one non-zero element at the first entry, $\underline{\mathbf{H}}_{i,E}\underline{\mathbf{x}}_E = x_{\text{pilot}}\underline{\mathbf{h}}_{i,E,1}$ where $\underline{\mathbf{h}}_{i,E,1}$ is the first column of $\underline{\mathbf{H}}_{i,E}$ and is thus dependent on $l_i$ and $\nu_i$ as can be seen from (33) and (37), i.e., $\underline{\mathbf{h}}_{i,E,1} = \underline{\mathbf{h}}_{i,E,1}(l_i, \nu_i)$ or equivalently $\underline{\mathbf{h}}_{i,E,1} = \underline{\mathbf{h}}_{i,E,1}(l_i, \alpha_i, a_i)$. Thus, the ML estimator is given by

$$
\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta} \in \mathbb{C}^P \times \mathbb{R}^P \times \mathbb{R}^P} \left\|\underline{\mathbf{y}}_E - x_{\text{pilot}} \sum_{i=0}^{P-1} h_i \underline{\mathbf{h}}_{i,E,1}(l_i, \nu_i)\right\|^2. \tag{75}
$$

> **Fig. 7.** Symbol arrangement at the transmitter.

> **Fig. 8.** Received frame at the receiver.

As brute force search is infeasible in a $3P$-dimensional continuous domain, we propose a low complexity solution to (75). For given $\{l_i, \nu_i\}$, the log-likelihood function in (74) is quadratic in the complex gain $h_i$. Therefore, solving (75) with respect to $h_i$ leads to the linear system of equations

$$
\sum_{j=0}^{P-1} h_j \underline{\mathbf{h}}_{i,E,1}^H(l_i, \nu_i) \underline{\mathbf{h}}_{j,E,1}(l_j, \nu_j) = \frac{\underline{\mathbf{h}}_{i,E,1}^H(l_i, \nu_i) \underline{\mathbf{y}}_E}{x_{\text{pilot}}}, \quad i = 0, 1, \cdots, P-1. \tag{76}
$$

Expanding (75) and using (76), the minimization with respect to the $\{l_i, \nu_i\}$ reduces to maximizing the function

$$
l_2(\underline{\mathbf{y}}_E | \boldsymbol{\theta}, x_{\text{pilot}}) = \sum_{i=0}^{P-1} \frac{|\underline{\mathbf{h}}_{i,E,1}^H(l_i, \nu_i)\underline{\mathbf{y}}_E|^2}{\underline{\mathbf{h}}_{i,E,1}^H(l_i, \nu_i)\underline{\mathbf{h}}_{i,E,1}(l_i, \nu_i)} - \frac{(\sum_{j \neq i} h_j \underline{\mathbf{h}}_{i,E,1}^H(l_i, \nu_i)\underline{\mathbf{h}}_{j,E,1}(l_j, \nu_j))\underline{\mathbf{y}}_E^H \underline{\mathbf{h}}_{i,E,1} x_{\text{pilot}}}{\underline{\mathbf{h}}_{i,E,1}^H(l_i, \nu_i)\underline{\mathbf{h}}_{i,E,1}(l_i, \nu_i)}. \tag{77}
$$

Now considering (76) and (77), we show how channel estimation is performed for the integer and fractional Doppler shift cases in the following subsections.

### A. Integer Doppler Case

In this case ($\nu_i = \alpha_i$), as it can be seen from (33), $\underline{\mathbf{h}}_{i,E,1}$ has only one non-zero element. In addition, for different paths, the location of these non-zero elements are different from each other, i.e.,

$$
\underline{\mathbf{h}}_{i,E,1}^H(l_i, \alpha_i) \underline{\mathbf{h}}_{j,E,1}(l_j, \alpha_j) = \begin{cases} 1, & i = j \\ 0, & i \neq j \end{cases}. \tag{78}
$$

Thus, (76) and (77) are rewritten as

$$
h_i = \frac{\underline{\mathbf{h}}_{i,E,1}^H(l_i, \alpha_i) \underline{\mathbf{y}}_E}{x_{\text{pilot}}}, \quad i = 0, 1, \cdots, P-1, \tag{79}
$$

$$
l_2(\underline{\mathbf{y}}_E | \boldsymbol{\theta}, x_{\text{pilot}}) = \sum_{i=0}^{P-1} |\underline{\mathbf{h}}_{i,E,1}^H(l_i, \alpha_i) \underline{\mathbf{y}}_E|^2, \tag{80}
$$

respectively. Thus, the delays and Doppler shifts, i.e., $\mathbf{l} \triangleq [l_0, \ldots, l_{P-1}]$ and $\boldsymbol{\alpha} \triangleq [\alpha_0, \ldots, \alpha_{P-1}]$ can be estimated as the argument maximizing the r.h.s. of (80). Due to the structure of $\underline{\mathbf{h}}_{i,E,1}(l_i, \alpha_i)$, maximizing the r.h.s. of (80) is equivalent to finding the indices of the largest entries of $\underline{\mathbf{y}}_E$. After finding the pair of parameters $\{l_i, \alpha_i\}$ for all the paths, the paths complex gains can be obtained using (79).

### B. Fractional Doppler Case

For the fractional case ($\nu_i = \alpha_i + a_i$), as it is shown in (37), (78) cannot in theory hold. Therefore, it is impossible to directly maximize (77) as the complex gains $h_i$ are not known. Moreover, the second term in (77) depends on all pairs of $\{l_i, \nu_i\}$ for $j \neq i$. However, assuming large enough $\xi_\nu$, the value of $\underline{\mathbf{h}}_{i,E,1}^H(l_i, \alpha_i, a_i)\underline{\mathbf{h}}_{j,E,1}(l_j, \alpha_j, a_j)$ is very small when $i \neq j$. Thus, for the fractional case, we exploit this approximation and assume that (78) holds to maximize (77). With this assumption, in order to find the $\{l_i, \nu_i\}$ pairs that maximize (77), first, we find the delay and integer part of the Doppler shift, i.e., $\mathbf{l}$ and $\boldsymbol{\alpha}$. To this end, we denote all the delays and integer Doppler shifts combinations set by $\mathcal{L} \triangleq \{(\mathbf{l}, \boldsymbol{\alpha}) | 0 \leq l[i] \leq l_{\max}, -\alpha_{\max} \leq \alpha[i] \leq \alpha_{\max}\}$ and pick the one that maximizes (77). In the next phase, the fractional parts $\mathbf{a} \triangleq [a_0, \ldots, a_{P-1}]$ are estimated using the obtained delay and integer Doppler shifts as

$$
\hat{\mathbf{a}} = \arg\max_{\mathbf{a}} \sum_{i=0}^{P-1} \frac{|\underline{\mathbf{h}}_{i,E,1}^H(\hat{l}_i, \hat{\alpha}_i, a_i)\underline{\mathbf{y}}_E|^2}{\underline{\mathbf{h}}_{i,E,1}^H(\hat{l}_i, \hat{\alpha}_i, a_i)\underline{\mathbf{h}}_{i,E,1}(\hat{l}_i, \hat{\alpha}_i, a_i)}, \tag{81}
$$

using a search on fine discretization of $[-1/2, 1/2]^P$. Then, the estimated Doppler shifts become $\hat{\boldsymbol{\nu}} \triangleq [\hat{\nu}_0, \ldots, \hat{\nu}_{P-1}] = \hat{\boldsymbol{\alpha}} + \hat{\mathbf{a}}$. The complex gains are estimated by solving the linear system (76). It is worth noting that this algorithm has good performance if the paths have different delays, i.e., for each delay, there is only one Doppler shift.

---

## VII. Simulation Results

In this section, we provide simulation results to assess the performance of AFDM. In all simulations, the complex gains $h_i$ are generated as independent complex Gaussian random variables with zero mean and $1/P$ variance. The carrier frequency is 4 GHz. BER values are obtained using $10^6$ different channel realizations.

Fig. 9a shows the simulated BER performance of AFDM for different channels with $N = 16$ and BPSK using the ML detection. We consider three different channels with different numbers of paths, namely a 2-path, a 3-path, and a 4-path channel. The maximum delay spread (in terms of integer taps) is set to be 2 ($l_{\max} = 1$), 3 ($l_{\max} = 2$) and 4 ($l_{\max} = 3$), respectively. The duration between two successive delay taps is approximately 41.6 µs. The maximum Doppler shift is considered $\alpha_{\max} = 1$, which corresponds to a maximum speed of 405 km/h. We observe that for each channel, AFDM achieves the optimal diversity order of the channel. Note that the plots of $s_1(\text{SNR})^{-2}$, $s_2(\text{SNR})^{-3}$ and $s_3(\text{SNR})^{-4}$ are only used to identify the slope of the curves and do not represent an upper bound.

> **Fig. 9.** BER performance using BPSK in LTV channels using ML detection. (a) BER performance of AFDM with different number of paths for $N = 16$. (b) BER performance of OFDM, OCDM, OTFS and AFDM in a three-path channel for $N = 16$, $N_{\text{OTFS}} = 4$ and $M_{\text{OTFS}} = 4$.

Before proceeding further, we recall that in OTFS the time–frequency signal plane is sampled at intervals $T_{\text{OTFS}}$ (seconds) and $\Delta f_{\text{OTFS}}$ (Hz), respectively to obtain a grid as

$$
\Lambda_{\text{OTFS}} = \{(nT_{\text{OTFS}}, m\Delta f_{\text{OTFS}}), \; n = 0, \cdots, N_{\text{OTFS}} - 1, \; m = 0, \cdots, M_{\text{OTFS}} - 1\}, \tag{82}
$$

and modulated time–frequency samples $X[n, m]$, $n = 0, \ldots, N_{\text{OTFS}} - 1$, $m = 0, \ldots, M_{\text{OTFS}} - 1$ are transmitted over an OTFS frame with duration $T_{f\text{-OTFS}} = N_{\text{OTFS}} T_{\text{OTFS}}$ and occupy a bandwidth $B_{\text{OTFS}} = M_{\text{OTFS}} \Delta f_{\text{OTFS}}$. In order to compare the performance of AFDM against OTFS, we assume $N = M_{\text{OTFS}} N_{\text{OTFS}}$ so that the AFDM and OTFS frames occupy the same time-frequency resources. We also compare the computational complexity of OTFS with AFDM. In the symplectic finite Fourier transform (SFFT) implementation of OTFS [28], the transmitter includes, in addition to OFDM modulation, an $\mathbf{F}_{M_{\text{OTFS}}}^H \otimes \mathbf{F}_{N_{\text{OTFS}}}$ step whose complexity is equivalent to $M_{\text{OTFS}} N_{\text{OTFS}}$-sized ($N$-sized) FFT, i.e., $(N/2)\log_2 N$ complex multiplications. For AFDM, there are two additional phase rotations, which require 2 additional complex multiplications per symbol compared to the transmitter of an OFDM system, which means the additional complexity is $2N$ complex multiplications. Table I summarizes this complexity comparison¹.

> ¹We compare AFDM only with SFFT-based OTFS, since the DZT-based OTFS [29] has limitations in terms of spectral shaping and compatibility with OFDM transceivers.

**TABLE I: Excess Complexity of OTFS and AFDM Transmitters Over OFDM Transmitter**

| Waveform | Complex Multiplication |
|:--------:|:----------------------:|
| OTFS     | $(N/2)\log_2 N$        |
| AFDM     | $2N$                   |

Fig. 9b shows the BER performance of AFDM, OFDM, OCDM, and OTFS. For the DAFT-based schemes, we generate the frames with $N = 16$. OTFS frame is generated with $N_{\text{OTFS}} = 4$ and $M_{\text{OTFS}} = 4$. The maximum delay spread is set to be $l_{\max} = 2$ and the maximum Doppler shift is $\alpha_{\max} = 1$. The delay shifts are fixed and Jakes Doppler spectrum is considered for each channel realization, i.e., the Doppler shifts are varying and the Doppler shift of each path is generated using $\alpha_i = \alpha_{\max}\cos(\theta_i)$, where $\theta_i$ is uniformly distributed over $[-\pi, \pi]$. Expectedly, OFDM has the worse performance as it cannot separate the paths. The performance of OCDM depends on the delay-Doppler profile of the channel. OCDM performs poorly and has the same diversity (one) as OFDM, due to the possible destructive addition of the two overlapping paths. The reason why OCDM has better performance than OFDM is related to its better path separation capabilities than OFDM. The proposed AFDM achieves the optimal diversity order, mainly due to path separation by tuning $c_1$ and setting $c_2$ to be an arbitrary irrational number or a rational number sufficiently smaller than $\frac{1}{2N}$. We also observe that AFDM has the same BER performance as OTFS. The reason AFDM and OTFS have (almost) the same BER performance is because AFDM and OTFS both achieve a full delay Doppler representation of the channel, i.e., of the paths of the effective channel (in the DAFT domain for AFDM, in the delay-Doppler domain for OTFS) each corresponds to one delay tap-Doppler bin pair of the wireless propagation channel and each has the same path gain under both AFDM and OTFS (strict equality holds at least in the case of integer valued Doppler shifts). As mentioned earlier, this feature enables AFDM to achieve the optimal diversity order of the LTV channels. While OTFS does not strictly speaking achieve the diversity order of the LTV channel, the number of codeword pairs with PEP that decays with slope 1 with respect to the SNR is too small to have an effect on the overall PEP in the intermediate SNR regime [21].

In the previous figures, small $N$ values are assumed along with ML detection to show the diversity order of AFDM. In the following figures, we consider a more practical configuration with QPSK, $N = 256$ for the DAFT-based schemes and $N_{\text{OTFS}} = 16$, $M_{\text{OTFS}} = 16$ for the OTFS. We consider a 3-path channel. The maximum Doppler shift is $\alpha_{\max} = 2$, which corresponds to a speed of 540 km/h, and the Doppler shift of each path is generated using Jakes Doppler spectrum. The maximum delay spread is set to be $l_{\max} = 2$. Fig. 10a shows the BER performance of the DAFT-based schemes and OTFS. All results are obtained with LMMSE detection at the receiver. We observe that AFDM outperforms OFDM and OCDM, while having identical performance with OTFS. However, when channel estimation is taken into account, the pilot overhead of OTFS is twice that of AFDM due to the 2D structure of its underlying transform. Indeed, while the AFDM embedded pilot scheme presented in Section VI occupies $2(l_{\max} + 1)(2(\alpha_{\max} + \xi_\nu) + 1) - 1$ entries out of the $N$ entries of the AFDM symbol, its OTFS counterpart [22] requires $(4(\alpha_{\max} + \xi_\nu) + 1)(2l_{\max} + 1)$ (for the integer Doppler shifts $\xi_\nu = 0$). This difference as shown in Fig. 11 translates into a significant advantage of AFDM over OTFS in terms of spectral efficiency, as shown in Fig. 10b. The spectral efficiency values were derived from the BER values plotted in Fig. 10a.

> **Fig. 10.** BER and spectral efficiency performance of OFDM, OCDM, OTFS and AFDM using MMSE detection. (a) BER. (b) Spectral efficiency.

> **Fig. 11.** Excess pilot guard overhead in OTFS with respect to AFDM.

Fig. 12 compares the performance of AFDM and OFDM in terms of BER using different detectors. In this figure, integer and fractional Doppler shifts are considered. We observe that AFDM outperforms OFDM, owed to achieving the optimal diversity order and every information symbol being received through multiple independent non-overlapping paths. Moreover, it shows that the weighted MRC-based DFE has close performance to exact LMMSE, which validates Theorem 2. In Fig. 12a, all BER curves (i.e., of the two methods and of low-complexity MMSE [2] based on banded matrix approximation) coincide because the channel matrices used in the three detection methods are all the same and are banded without approximation. In Fig. 12b, exact LMMSE has slightly better performance than the low-complexity methods due to the use of the banded-matrix approximation in the latter when Doppler frequency shifts are fractional.

> **Fig. 12.** BER performance comparison between AFDM and OFDM systems using different detectors for the integer and fractional Doppler shifts. (a) Integer Doppler shifts. (b) Fractional Doppler shifts.

We now assess the BER performance of AFDM when detection is performed based on the channel state information given by the proposed channel estimation scheme. The pilot symbol SNR is denoted by $\text{SNR}_p = \frac{|x_{\text{pilot}}|^2}{N_0}$ and the data symbols have the $\text{SNR}_d = \frac{\mathbb{E}\{|x^{\text{data}}|^2\}}{N_0}$.

Fig. 13 shows the BER versus $\text{SNR}_d$ for AFDM considering the ideal case of perfect channel knowledge at the receiver as well as the case where the channel is estimated using the proposed algorithm for integer Doppler case with different values of $\text{SNR}_p$. As $\text{SNR}_p$ increases, the BER decreases and the AFDM performance improves. Moreover, we see that for $\text{SNR}_p = 35$ dB, the performance of AFDM with the proposed channel estimation is very close to the ideal case.

> **Fig. 13.** BER versus $\text{SNR}_d$ for the integer Doppler case with different $\text{SNR}_p$ and ideal channel.

Fig. 14a shows the BER performance of AFDM for different $\text{SNR}_p$ considering the fractional Doppler shift case. Similar to the integer Doppler shift case, increasing the pilot power improves the error performance. As we can see, with $\text{SNR}_p = 40$ dB, AFDM with the proposed embedded channel estimation has similar performance with AFDM with perfect channel knowledge at the receiver. Note that the system has more overhead in the fractional Doppler shift case. In addition, larger $\text{SNR}_p$ is needed to achieve the same performance. Note that in practice, it is possible to assume larger values for $\text{SNR}_p$ compared to $\text{SNR}_d$ since the zero guard samples surrounding the pilot symbol allow for the transmit power of the latter to be boosted without violating the average transmit power constraint. Under ideal receiver-side channel knowledge, it can be seen from Fig. 14b that increasing $\xi_\nu$, improves the performance of AFDM, as less overlapping is occurring between the matrices $\mathbf{H}_i$ belonging to different channel paths in the effective channel matrix $\mathbf{H}_{\text{eff}}$. With practical channel estimation, Fig. 14b shows that increasing $\xi_\nu$ also improves the channel estimation quality, since inter-path interference when channel estimation algorithm for fractional Doppler shift case is performed decreases for the same reason, i.e., less overlapping between the matrices $\mathbf{H}_i$.

> **Fig. 14.** The effect of $\text{SNR}_p$ and $\xi_\nu$ on the BER performance. (a) BER versus $\text{SNR}_d$ considering different $\text{SNR}_p$ and ideal receiver-side CSI. (b) BER versus $SNR_d$ with different $\xi_\nu$ considering ideal and estimated channel with $\text{SNR}_p = 40$ dB.

---

## VIII. Conclusion

We proposed a new waveform, coined AFDM, which employs multiple discrete-time orthogonal chirp signals generated using the discrete affine Fourier transform. The unique features and effects of DAFT were revealed by deriving the input-output relation. Using the input-output relation, the AFDM parameters can be tuned such that the DAFT domain channel impulse response constitutes a full representation of its delay-Doppler profile. Then, we showed analytically that AFDM can achieve the optimal diversity order in doubly dispersive channel by properly tuning its pulse parameters. Inserting zero-padding in the DAFT domain, we proposed a low complexity detector and channel estimation algorithms for AFDM. Simulation results showed that AFDM outperforms OFDM and other DAFT-based multicarrier schemes, while having advantages over OTFS in terms of pilot and user multiplexing overhead. The main takeaway of this paper is that AFDM is a promising new waveform for high mobility communications in future wireless systems.

---

## Appendix A: Proof of Theorem 1

We give the proof of Theorem 1 only in the case of integer Doppler shifts. The proof holds also for the fractional Doppler shift with some modifications.

First, we show that when (47) and (56) hold, there exist values of $c_2$ such that the rank of $\boldsymbol{\Phi}(\boldsymbol{\delta})$ is equal to $P$, i.e., such that the $P$ columns of $\boldsymbol{\Phi}(\boldsymbol{\delta})$ are linearly independent. Therefore, considering the $\boldsymbol{\Phi}(\boldsymbol{\delta})$ for a $P$-path channel:

$$
\boldsymbol{\Phi}(\boldsymbol{\delta}) = [\mathbf{H}_1\boldsymbol{\delta} \;|\; \ldots \;|\; \mathbf{H}_P\boldsymbol{\delta}] = \begin{pmatrix} H_{eff}[0, \text{loc}_1]\delta[\text{loc}_1] & \cdots & H_{eff}[0, \text{loc}_P]\delta[\text{loc}_P] \\ H_{eff}[1, (\text{loc}_1+1)_N]\delta[(\text{loc}_1+1)_N] & \cdots & H_{eff}[1, (\text{loc}_P+1)_N]\delta[(\text{loc}_P+1)_N] \\ \vdots & & \vdots \\ H_{eff}[N-1, (\text{loc}_1+N-1)_N]\delta[(\text{loc}_1+N-1)_N] & \cdots & H_{eff}[N-1, (\text{loc}_P+N-1)_N]\delta[(\text{loc}_P+N-1)_N] \end{pmatrix} \tag{83}
$$

we should show that

$$
\beta_1 \mathbf{H}_1 \boldsymbol{\delta} + \beta_2 \mathbf{H}_2 \boldsymbol{\delta} + \cdots + \beta_P \mathbf{H}_P \boldsymbol{\delta} = \mathbf{0} \implies \beta_1 = \beta_2 = \cdots = \beta_P = 0, \tag{84}
$$

which is proved by contradiction. Assume that there is at least one $\beta_i \neq 0$ and $\beta_1 \mathbf{H}_1 \boldsymbol{\delta} + \beta_2 \mathbf{H}_2 \boldsymbol{\delta} + \cdots + \beta_P \mathbf{H}_P \boldsymbol{\delta} = \mathbf{0}$. Without loss of generality (wlog), we assume $\beta_1 \neq 0$. Dividing both sides of the vector equality in (84) by $\beta_1$ and considering the first entry of the resulting vector, we have

$$
\delta[\text{loc}_1] = -\frac{H[0, \text{loc}_2]}{H[0, \text{loc}_1]} \frac{\beta_2}{\beta_1} \delta[\text{loc}_2] - \cdots - \frac{H[0, \text{loc}_P]}{H[0, \text{loc}_1]} \frac{\beta_P}{\beta_1} \delta[\text{loc}_P]. \tag{85}
$$

In addition, by taking into account the $H_{eff}$ expression, we have

$$
\frac{H[0, \text{loc}_i]}{H[0, \text{loc}_j]} = e^{\imath 2\pi c_2(\text{loc}_i^2 - \text{loc}_j^2)} e^{\imath \frac{2\pi}{N}(Nc_1(l_i^2 - l_j^2) - \text{loc}_i l_i + \text{loc}_j l_j)}. \tag{86}
$$

Now we can rewrite (85) using (86)

$$
\delta[\text{loc}_1] = e^{-\imath 2\pi c_2 \text{loc}_1^2} e^{\imath \frac{2\pi}{N}(Nc_1(-l_1^2 + \text{loc}_1 l_1))} \times \sum_{i=2}^{P} e^{\imath 2\pi c_2 \text{loc}_i^2} e^{\imath \frac{2\pi}{N}(Nc_1 l_i^2 - \text{loc}_i l_i)} \beta'_i \delta[q_i], \tag{87}
$$

where $\beta'_i = \frac{-\beta_i}{\beta_1}$. Note that $\boldsymbol{\delta} \in \mathbb{Z}[j]^{N \times 1}$, therefore since $\delta[\text{loc}_1] \in \mathbb{Z}[j]$, then the r.h.s. of (87) should be in $\mathbb{Z}[j]$ to have the equality. On the other hand, choosing an irrational number for $c_2$, then $e^{\imath 2\pi c_2 q_i^2}$ is an irrational number and since (87) should hold for different values of $\boldsymbol{\delta}$, the effect of $c_2$ should be removed from this equation. This can be done by choosing

$$
\beta'_i = e^{\imath 2\pi c_2 q_1^2} e^{-\imath 2\pi c_2 q_i^2} \mu_i, \quad i = 2, \cdots, P \tag{88}
$$

where $\mu_i$s do not contain $c_2$ in their phases. Now in order to have (87) hold, at least another $\beta_i$ should be non-zero. Again wlog, assume the non-zero one is $\beta_2$. Dividing both sides of the vector equality in (84) with $\beta_2$ and considering the second entry of the resulting vector we get

$$
\delta[(\text{loc}_2 + 1)_N] = e^{-\imath 2\pi c_2(\text{loc}_2 + 1)_N^2} e^{\imath \frac{2\pi}{N}(Nc_1(-l_2^2 + (\text{loc}_2+1)_N l_2))} \times \sum_{i=1, i\neq 2}^{P} e^{\imath 2\pi c_2(\text{loc}_i + 1)_N^2} \times e^{\imath \frac{2\pi}{N}(Nc_1 l_i^2 - (\text{loc}_i + 1)_N l_i)} \beta''_i \delta[(\text{loc}_i + 1)_N], \tag{89}
$$

where $\beta''_i = \frac{-\beta_i}{\beta_2}$. With the same explanation, $\beta''_i$s are

$$
\beta''_i = e^{\imath 2\pi c_2(q_2 + 1)_N^2} e^{-\imath 2\pi c_2(q_i + 1)_N^2} \mu'_i, \quad i = 1, \cdots, P, \; i \neq 2. \tag{90}
$$

Putting (88) and (90) together shows

$$
\beta'_2 = \frac{1}{\beta''_1}, \tag{91}
$$

which leads to

$$
e^{\imath 2\pi c_2(\text{loc}_2^2 - (\text{loc}_2 + 1)_N^2 + \text{loc}_1^2 - (\text{loc}_1 + 1)_N^2)} \mu_2 \mu'_1 = 1. \tag{92}
$$

On one hand, since $\text{loc}_1 \neq \text{loc}_2$, $\text{loc}_2^2 - (\text{loc}_2 + 1)_N^2 + \text{loc}_1^2 - (\text{loc}_1 + 1)_N^2$ cannot be zero and on the other hand, we know that $\mu_i$ and $\mu'_i$ do not contain $c_2$ in their phases. Therefore the $c_2$ effect in the phase cannot be removed and the l.h.s. of (92) is an irrational number while the right one is integer. Thus, our initial assumption does not hold and $\beta_1 = \cdots = \beta_P = 0$, which means that the $P$ columns of $\boldsymbol{\Phi}(\boldsymbol{\delta})$ are linearly independent, i.e., the rank of $\boldsymbol{\Phi}(\boldsymbol{\delta})$ is $P$. ∎

---

## Appendix B: Proof of Theorem 2

In order to prove the convergence, we should show that $|\lambda(-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S}))| < 1$, where $\lambda$ denotes any eigenvalue of $-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S})$. For the corresponding eigenvector $\mathbf{v}$, where $\mathbf{v}^H \mathbf{v} = \beta > 0$, we can write

$$
-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S})\mathbf{v} = \lambda(-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S}))\mathbf{v}. \tag{93}
$$

After multiplying both sides of (93) by $\mathbf{v}^H \mathbf{S}$, it writes as

$$
\lambda(-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S})) = \frac{\mathbf{v}^H(-(\mathbf{R} - \mathbf{S}))\mathbf{v}}{\mathbf{v}^H \mathbf{S} \mathbf{v}}. \tag{94}
$$

Considering $\mathbf{S} = (\gamma^{-1} + d)\mathbf{I} + \mathbf{L}$ and $\mathbf{R} = \underline{\mathbf{H}}_{\text{eff}}^H \underline{\mathbf{H}}_{\text{eff}} + \gamma^{-1}\mathbf{I} = \mathbf{L} + \mathbf{L}^H + (d + \gamma^{-1})\mathbf{I}$, (94) becomes

$$
\lambda(-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S})) = \frac{-\mathbf{v}^H \mathbf{L}^H \mathbf{v}}{(\gamma^{-1} + d)\mathbf{v}^H \mathbf{v} + \mathbf{v}^H \mathbf{L} \mathbf{v}}. \tag{95}
$$

Since $\mathbf{R}$ is positive definite Hermitian matrix, any non-zero vector including $\mathbf{v}$ satisfies

$$
\mathbf{v}^H(\underline{\mathbf{H}}_{\text{eff}}^H \underline{\mathbf{H}}_{\text{eff}} + \gamma^{-1}\mathbf{I})\mathbf{v} = \mathbf{v}^H(\mathbf{L} + \mathbf{L}^H + (d + \gamma^{-1})\mathbf{I})\mathbf{v} = \beta(d + \gamma^{-1}) + 2\Re(\mathbf{v}^H \mathbf{L} \mathbf{v}) > 0, \tag{96}
$$

where (96) can be written as

$$
a = \Re(\mathbf{v}^H \mathbf{L} \mathbf{v}) = \Re(\mathbf{v}^H \mathbf{L}^H \mathbf{v}) > \frac{-\beta(d + \gamma^{-1})}{2}, \tag{97}
$$

and the imaginary part is equal to

$$
b = \Im(\mathbf{v}^H \mathbf{L} \mathbf{v}) = -\Im(\mathbf{v}^H \mathbf{L}^H \mathbf{v}). \tag{98}
$$

Therefore, (94) can be rewritten as

$$
|\lambda(-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S}))| = \frac{|a - \imath b|}{|\beta(\gamma^{-1} + d) + a - \imath b|}. \tag{99}
$$

From (97) and (99), it can be seen that $|\lambda(-\mathbf{S}^{-1}(\mathbf{R} - \mathbf{S}))| < 1$. ∎

---

## References

[1] A. Bemani, N. Ksairi, and M. Kountouris, "AFDM: A full diversity next generation waveform for high mobility communications," in *Proc. IEEE Int. Conf. Commun. Workshops (ICC Workshops)*, Jun. 2021, pp. 1–6.

[2] A. Bemani, N. Ksairi, and M. Kountouris, "Low complexity equalization for AFDM in doubly dispersive channels," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, May 2022, pp. 1–5.

[3] T. Wang, J. G. Proakis, E. Masry, and J. R. Zeidler, "Performance degradation of OFDM systems due to Doppler spreading," *IEEE Trans. Wireless Commun.*, vol. 5, no. 6, pp. 1422–1432, Jun. 2006.

[4] "The industrial reorganization act: The communications industry," *Proc. Inst. Electr. Engineers*, vol. 73, no. 3, 1973, pp. 635–676. [Online]. Available: https://www.jstor.org/stable/1121267

[5] G. Gott and J. Newsome, "HF data transmission using chirp signals," *Proc. Inst. Elect. Eng.*, vol. 118, no. 9, 1971, pp. 1162–1166.

[6] A. Kadri, R. K. Rao, and J. Jiang, "Low-power chirp spread spectrum signals for wireless communication within nuclear power plants," *Nucl. Technol.*, vol. 166, no. 2, pp. 156–169, May 2009.

[7] C. He, M. Ran, Q. Meng, and J. Huang, "Underwater acoustic communications using M-ary chirp-DPSK modulation," in *Proc. IEEE 10th Int. Conf. Signal Process.*, Oct. 2010, pp. 1544–1547.

[8] M. Palmese, G. Bertolotto, A. Pescetto, and A. Trucco, "Experimental validation of a chirp-based underwater acoustic communication method," in *Proc. Meetings Acoust.*, vol. 4, no. 1, 2008, Art. no. 030002.

[9] *IEEE standard for Information Technology—Local and Metropolitan Area Networks—Specific Requirements—Part 15.4: Wireless Medium Access Control (MAC) and Physical Layer (PHY) Specifications for Low-Rate Wireless Personal Area Networks (WPANs): Amendment 1: Add Alternate PHYs*, IEEE Standard 802.15.4a-2007, IEEE Standard 802.15.4-2006, 2007, pp. 1–210.

[10] M. Martone, "A multicarrier system based on the fractional Fourier transform for time-frequency-selective channels," *IEEE Trans. Commun.*, vol. 49, no. 6, pp. 1011–1020, Jun. 2001.

[11] T. Erseghe, N. Laurenti, and V. Cellini, "A multicarrier architecture based upon the affine Fourier transform," *IEEE Trans. Commun.*, vol. 53, no. 5, pp. 853–862, May 2005.

[12] D. Stojanović, I. Djurović, and B. R. Vojcic, "Multicarrier communications based on the affine Fourier transform in doubly-dispersive channels," *EURASIP J. Wireless Commun. Netw.*, vol. 2010, no. 1, pp. 1–10, Dec. 2010.

[13] X. Ouyang and J. Zhao, "Orthogonal chirp division multiplexing," *IEEE Trans. Commun.*, vol. 64, no. 4, pp. 3946–3957, Sep. 2016.

[14] M. S. Omar and X. Ma, "Performance analysis of OCDM for wireless communications," *IEEE Trans. Wireless Commun.*, vol. 20, no. 7, pp. 4032–4043, Jul. 2021.

[15] H. G. Myung, J. Lim, and D. J. Goodman, "Single carrier FDMA for uplink wireless transmission," *IEEE Veh. Technol. Mag.*, vol. 1, no. 3, pp. 30–38, Sep. 2006.

[16] G. Fettweis, M. Krondorf, and S. Bittner, "GFDM—Generalized frequency division multiplexing," in *Proc. VTC Spring IEEE 69th Veh. Technol. Conf.*, Apr. 2009, pp. 1–4.

[17] N. Michailow *et al.*, "Generalized frequency division multiplexing for 5th generation cellular networks," *IEEE Trans. Commun.*, vol. 62, no. 9, pp. 3045–3061, Sep. 2014.

[18] R. Hadani *et al.*, "Orthogonal time frequency space modulation," in *Proc. IEEE Wireless Commun. Netw. Conf. (WCNC)*, Mar. 2017, pp. 1–6.

[19] R. Hadani and A. Monk, "OTFS: A new generation of modulation addressing the challenges of 5G," 2018, arXiv:1802.02623.

[20] W. Anwar, A. Krause, A. Kumar, N. Franchi, and G. P. Fettweis, "Performance analysis of various waveforms and coding schemes in V2X communication scenarios," in *Proc. IEEE Wireless Commun. Netw. Conf. (WCNC)*, May 2020, pp. 1–8.

[21] G. D. Surabhi, R. M. Augustine, and A. Chockalingam, "On the diversity of uncoded OTFS modulation in doubly-dispersive channels," *IEEE Trans. Wireless Commun.*, vol. 18, no. 6, pp. 3049–3063, Jun. 2019.

[22] P. Raviteja, K. T. Phan, and Y. Hong, "Embedded pilot-aided channel estimation for OTFS in delay–Doppler channels," *IEEE Trans. Veh. Tech.*, vol. 68, no. 5, pp. 4906–4917, May 2019.

[23] A. Bemani, G. Cuozzo, N. Ksairi, and M. Kountouris, "Affine frequency division multiplexing for next-generation wireless networks," in *Proc. 17th Int. Symp. Wireless Commun. Syst. (ISWCS)*, Sep. 2021, pp. 1–6.

[24] J. J. Healy, M. A. Kutay, H. M. Ozaktas, and J. T. Sheridan, *Linear Canonical Transforms: Theory and Applications*, vol. 198. Cham, Switzerland: Springer, 2015.

[25] S.-C. Pei and J.-J. Ding, "Closed-form discrete fractional and affine Fourier transforms," *IEEE Trans. Signal Process.*, vol. 48, no. 5, pp. 1338–1353, May 2000.

[26] Å. Björck, *Numerical Methods for Least Squares Problems*. Philadelphia, PA, USA: SIAM, 1996.

[27] Y. Saad, *Iterative Methods for Sparse Linear Systems*. Philadelphia, PA, USA: SIAM, 2003.

[28] Z. Wei, S. Li, W. Yuan, R. Schober, and G. Caire, "Orthogonal time frequency space modulation—Part I: Fundamentals and challenges ahead," 2022, arXiv:2209.05011.

[29] Z. Wei, S. Li, W. Yuan, R. Schober, and G. Caire, "Orthogonal time frequency space modulation—Part I: Fundamentals and challenges ahead," *IEEE Commun. Lett.*, vol. 27, no. 1, pp. 4–8, Jan. 2022.

---

## Author Biographies

**Ali Bemani** (Member, IEEE) received the B.Sc. degree in electrical engineering from the Amirkabir University of Technology, Tehran, Iran, in 2015, and the M.Sc. degree in communication systems from the Sharif University of Technology, Tehran, in 2017. He is currently pursuing the Ph.D. degree with Sorbonne University, France. He is also a Research Engineer at the Mathematical and Algorithmic Sciences Laboratory, Huawei France Research and Development, Paris, France. Previously, he was a Researcher at EURECOM, Sophia-Antipolis, France. His research interests include wireless communications, waveform design, channel estimation, and integrated sensing and communications.

**Nassar Ksairi** (Senior Member, IEEE) received the M.Sc. degree from CentraleSupélec, France, in 2006, and the Ph.D. degree from the University of Paris-Sud XI, France, in 2010. From 2010 to 2012, he was an Assistant Professor at the Higher Institute for Applied Sciences and Technology, Damascus, Syria. From 2012 to 2014, he was a Post-Doctoral Researcher at Télécom ParisTech, Paris, France. Since December 2014, he has been a Researcher at the Mathematical and Algorithmic Sciences Laboratory, Huawei France Research and Development, France. His current research interests include wireless waveform design, channel estimation, and integrated sensing and communications.

**Marios Kountouris** (Fellow, IEEE) received the Diploma degree in electrical and computer engineering from the National Technical University of Athens (NTUA), Greece, in 2002, and the M.S. and Ph.D. degrees in electrical engineering from Télécom Paris, France, in 2004 and 2008, respectively. He is currently a Professor at the Communication Systems Department, EURECOM, France. Prior to his current appointment, he has held positions at CentraleSupélec, France, The University of Texas at Austin, Austin, TX, USA, the Huawei Paris Research Center, France, and Yonsei University, South Korea. He is a fellow of AAAI and a Professional Engineer of the Technical Chamber of Greece. He was a recipient of a Consolidator Grant of the European Research Council (ERC) in 2020 on goal-oriented semantic communication. He has received several awards and distinctions, including the 2022 Blondel Medal, the 2020 IEEE ComSoc Young Author Best Paper Award, the 2016 IEEE ComSoc CTTC Early Achievement Award, the 2013 IEEE ComSoc Outstanding Young Researcher Award for the EMEA Region, the 2012 IEEE SPS Signal Processing Magazine Award, the IEEE SPAWC 2013 Best Paper Award, and the IEEE Globecom 2009 Communication Theory Best Paper Award. He has served as an Editor for the IEEE Transactions on Wireless Communications, the IEEE Transactions on Signal Processing, and the IEEE Wireless Communications Letters.
