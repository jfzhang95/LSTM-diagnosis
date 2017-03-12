## LSTM-diagnosis
We build LSTM model to learn time series data gained from ICU:


1. DATA SOURCE: [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/)
2. 用softmax函数把输出转为一个 vector:[p1,p2,p3,....,p9], 是每种病患病的概率.
3. iters是指总循环次数,每次循环会把所有病人都循环一次.
