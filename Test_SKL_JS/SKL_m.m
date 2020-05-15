song1: music/Ooby_Dooby/roy_orbison+Black_and_White_Night+05-Ooby_Doobymp3
song2: music/Walk_This_Way/run_dmc+Raising_Hell+04-Walk_This_Waymp3: 

function [g] = mvn_new(cov, m)
    g.m = m(:);
    g.cov = cov;
    if (rcond(cov) < 1e-15)
        error("Covariance is badly scaled!");
    end
    g_chol = chol(cov);
    g.logdet = 2*sum(log(diag(g_chol)));
    g_ui = g_chol\eye(length(g.m));
    g.icov = g_ui*g_ui';
endfunction

song1_m = [-692.95306  , 173.50534  , -51.494907 ,  55.741478 , -23.526962 ,   29.248734 , -16.82424  ,   5.936065 ,   2.7149098,  -4.229291 ,   13.672986 , -11.684442 ,  12.167524 ];
song1_m = song1_m(:); 
song1_cov = [[2941.24307693,   8.52134777, -75.57629659,1109.20320609,-515.94068915,   736.53946892,-187.14399948, 128.16572867,  12.11674654,-135.80967485,   155.84429367,-221.73978295, 207.15935782]; 
[   8.52134777, 653.64827643, 310.99184394,-185.72470222, 185.84253085,     2.1053451 ,  50.17772801, -32.60113656,  22.16596073,  22.98229825,   -14.85181814,  94.03251853, -33.18497561]; 
[ -75.57629659, 310.99184394, 668.12811691, 167.19645432, -29.82811494,   168.64070845,  23.55994865,  12.62516653,  24.47342807, -15.73260165,    28.50741512,  42.51211211,  19.54680639]; [1109.20320609,-185.72470222, 167.19645432, 748.03470884,-252.18142685,   319.63283697, -77.34307415,  68.77448108, -13.52012257, -63.92064998,    78.45782718,-122.69256906, 109.7139559 ]; 
[-515.94068915, 185.84253085, -29.82811494,-252.18142685, 317.76113661,  -140.7698069 ,  20.99754698, -27.76050373,  11.32520402,  49.53625548,   -39.30072121,  75.08442761, -41.41123531]; [ 736.53946892,   2.1053451 , 168.64070845, 319.63283697,-140.7698069 ,   442.86609236, -16.56201584,  53.87941947,  94.76741547, -50.65362989,    70.54971896, -19.18291825,  72.50146314]; 
[-187.14399948,  50.17772801,  23.55994865, -77.34307415,  20.99754698,   -16.56201584, 131.16179857,  19.15309988,  15.74894525,  25.12867131,    -3.47036704,  34.42172105, -20.37852948]; 
[ 128.16572867, -32.60113656,  12.62516653,  68.77448108, -27.76050373,    53.87941947,  19.15309988,  91.30653366,  31.97596717,  -5.91513031,    28.8848414 ,  -4.06059106,  22.02341938]; 
[  12.11674654,  22.16596073,  24.47342807, -13.52012257,  11.32520402,    94.76741547,  15.74894525,  31.97596717, 127.7838041 ,  12.50831894,    11.08933425,  40.93345385,   0.04366327]; 
[-135.80967485,  22.98229825, -15.73260165, -63.92064998,  49.53625548,   -50.65362989,  25.12867131,  -5.91513031,  12.50831894,  79.05256245,     9.50750361,   6.0551683 ,   3.36514014]; 
[ 155.84429367, -14.85181814,  28.50741512,  78.45782718, -39.30072121,    70.54971896,  -3.47036704,  28.8848414 ,  11.08933425,   9.50750361,    71.34167676,  12.17204216,  14.77289085]; 
[-221.73978295,  94.03251853,  42.51211211,-122.69256906,  75.08442761,   -19.18291825,  34.42172105,  -4.06059106,  40.93345385,   6.0551683 ,    12.17204216,  98.37829682,  -4.41269769]; 
[ 207.15935782, -33.18497561,  19.54680639, 109.7139559 , -41.41123531,    72.50146314, -20.37852948,  22.02341938,   0.04366327,   3.36514014,    14.77289085,  -4.41269769,  80.09894953]]
song1 = mvn_new(song1_cov, song1_m)
song2_m = [-684.114   , 150.25163 , -57.20039 ,  74.55266 , -31.272036,  26.839827,  -22.842867,   9.103122,   6.711247,  -8.166785,  16.002844, -15.141054,    9.713406] 
song2_m = song2_m(:); 
song2_cov = [[ 7421.72159519, 1751.3042376 ,-1800.27920557,  581.69385292,  -1002.75691919,  212.19862689, -638.12227247,   20.97453644,   -144.77701433, -286.46632609,  210.211626  , -303.39048869,    166.76280776]; [ 1751.3042376 , 1126.24885401, -377.04730618,   -6.01573473,     47.77935279, -238.70252974,    6.63891183,  -39.53990607,      0.7387623 ,  -16.85685899,   11.35273754,   71.52918922,    -48.37985038]; [-1800.27920557, -377.04730618,  854.9440749 ,   94.78071757,    283.23964162,  147.70050237,  183.0923488 ,   82.16560937,     67.1430082 ,   80.96981994,  -24.74579482,   54.55787516,     15.38834495]; [  581.69385292,   -6.01573473,   94.78071757,  376.9886836 ,      2.27718029,  202.67626536,  -28.75247726,   69.45455694,     29.96794928,   -5.380641  ,   41.09022681,  -47.79040322,     74.6519908 ]; [-1002.75691919,   47.77935279,  283.23964162,    2.27718029,    467.50279744,  -59.12866403,  195.51651681,   21.71779687,     75.06563765,  100.34857429,  -37.72104013,  118.08412685,    -29.95180657]; [  212.19862689, -238.70252974,  147.70050237,  202.67626536,    -59.12866403,  395.95636468,   14.14591316,  100.4281625 ,     17.83605485,   -9.73290974,   46.05238885,  -84.97696339,     87.45348139]; [ -638.12227247,    6.63891183,  183.0923488 ,  -28.75247726,    195.51651681,   14.14591316,  265.53186335,   56.86698865,     49.95287731,   68.52566115,  -16.3195532 ,   70.92423425,    -34.1532779 ]; [   20.97453644,  -39.53990607,   82.16560937,   69.45455694,     21.71779687,  100.4281625 ,   56.86698865,  140.52457798,     70.28064627,   31.48962438,   32.47587595,  -10.60429252,     29.00115903]; [ -144.77701433,    0.7387623 ,   67.1430082 ,   29.96794928,     75.06563765,   17.83605485,   49.95287731,   70.28064627,    122.01868414,   72.83238068,   14.81710683,   26.48036595,     -7.3927604 ]; [ -286.46632609,  -16.85685899,   80.96981994,   -5.380641  ,    100.34857429,   -9.73290974,   68.52566115,   31.48962438,     72.83238068,  112.01834601,   35.35223671,   33.34980499,    -11.14764727]; [  210.211626  ,   11.35273754,  -24.74579482,   41.09022681,    -37.72104013,   46.05238885,  -16.3195532 ,   32.47587595,     14.81710683,   35.35223671,   92.47305198,   24.67676151,     20.13664491]; [ -303.39048869,   71.52918922,   54.55787516,  -47.79040322,    118.08412685,  -84.97696339,   70.92423425,  -10.60429252,     26.48036595,   33.34980499,   24.67676151,  128.16161826,      9.04335851]; [  166.76280776,  -48.37985038,   15.38834495,   74.6519908 ,    -29.95180657,   87.45348139,  -34.1532779 ,   29.00115903,     -7.3927604 ,  -11.14764727,   20.13664491,    9.04335851,    100.01395676]]
song2 = mvn_new(song2_cov, song2_m)

d = mvn_div_js(song1, song2)
d = mvn_div_skl(song1, song2)

function d = my_mvn_div_js(m1, m2)
    m12.m = 0.5*m1.m + 0.5*m2.m; 
    t1 = 0.5*(m1.cov + m1.m*m1.m')
    t2 = 0.5*(m2.cov + m2.m*m2.m')
    t3 = m12.m*m12.m'
    m12.cov = t1 + t2 - t3;
    m12_chol = chol(m12.cov);
    m12.logdet = 2*sum(log(diag(m12_chol))) 
    d = 0.5*m12.logdet - 0.25*m1.logdet - 0.25*m2.logdet;
    d = max(d, 0);
end


m_mean = 0.5 * song1_m + 0.5 * song2_m;
m_cov = 0.5*(song1_cov + song1_m*song1_m') + 0.5*(song2_cov + song2_m*song2_m')- m_mean*m_mean';
m_cov_chol = chol(m_cov);
m_logdet = 2*sum(log(diag(m_cov_chol)))

d = 0.5*m_logdet - 0.25*song1.logdet - 0.25*song2.logdet

song2_m = [-1257.8583    ,    5.7125087 ,    2.0875356 ,    0.04315059,    -1.7997603 ,   -2.6178255 ,   -2.6397953 ,   -1.8498207 ,    -0.7610296 ,    0.1747297 ,    0.60117364,    0.61780185,     0.3042859 ] 
song2_m = song2_m(:); 
song2_cov = [[1343.08210732, 732.76709045,  43.54641283, -24.04915172,-101.41778303,   -76.2346755 , -72.84175518, -40.05904255, -11.22248678,  12.45501718,    -3.45919128, -12.70639546, -23.66364871]; 
[ 732.76709045, 420.08866473,  44.13072398,  -9.97146611, -66.64523978,   -61.63317823, -58.94587891, -35.69155546, -12.96368843,   6.55326272,     1.25486856,  -4.98161765, -12.77461102]; 
[  43.54641283,  44.13072398,  22.07742075,   2.72971167, -14.26345299,   -22.33562855, -21.62707948, -15.31334917,  -7.45168139,  -0.22428749,     2.63674573,   1.26171742,  -0.87703649]; 
[ -24.04915172,  -9.97146611,   2.72971167,   1.76900382,   1.22845608,    -0.65342145,  -1.06828181,  -1.43382666,  -1.39576969,  -1.19032453,    -0.47674433,  -0.28737755,  -0.12674202]; 
[-101.41778303, -66.64523978, -14.26345299,   1.22845608,  15.72077697,    18.71546625,  17.3761676 ,  10.94129726,   3.99696472,  -2.05953863,    -2.96713451,  -1.39780247,   0.80967876]; 
[ -76.2346755 , -61.63317823, -22.33562855,  -0.65342145,  18.71546625,    26.26303941,  25.03434349,  16.8432384 ,   7.04043277,  -1.57080417,    -4.49183957,  -2.95488359,  -0.05145455]; 
[ -72.84175518, -58.94587891, -21.62707948,  -1.06828181,  17.3761676 ,    25.03434349,  24.65939603,  17.0850153 ,   7.40355452,  -0.91398992,    -4.01771229,  -3.16167253,  -0.32271769]; 
[ -40.05904255, -35.69155546, -15.31334917,  -1.43382666,  10.94129726,    16.8432384 ,  17.0850153 ,  12.38071222,   5.84657714,   0.09447053,    -2.47949909,  -2.34224174,  -0.61269997]; 
[ -11.22248678, -12.96368843,  -7.45168139,  -1.39576969,   3.99696472,     7.04043277,   7.40355452,   5.84657714,   3.3920599 ,   0.91907688,    -0.4694769 ,  -0.63350746,  -0.27961411]; [  12.45501718,   6.55326272,  -0.22428749,  -1.19032453,  -2.05953863,    -1.57080417,  -0.91398992,   0.09447053,   0.91907688,   1.50119319,     1.33771472,   0.86307957,   0.31341065]; [  -3.45919128,   1.25486856,   2.63674573,  -0.47674433,  -2.96713451,    -4.49183957,  -4.01771229,  -2.47949909,  -0.4694769 ,   1.33771472,     2.3924506 ,   2.25984837,   1.55608972]; 
[ -12.70639546,  -4.98161765,   1.26171742,  -0.28737755,  -1.39780247,    -2.95488359,  -3.16167253,  -2.34224174,  -0.63350746,   0.86307957,     2.25984837,   2.87625505,   2.40974812]; 
[ -23.66364871, -12.77461102,  -0.87703649,  -0.12674202,   0.80967876,    -0.05145455,  -0.32271769,  -0.61269997,  -0.27961411,   0.31341065,     1.55608972,   2.40974812,   2.62378009]]
song2 = mvn_new(song2_cov, song2_m)

