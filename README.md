# NdLinear × Speech-Commands Benchmark  
_Compressing DS-CNN for keyword spotting without sacrificing accuracy_

## 1 • Motivation
Embedded keyword-spotting models run on battery-powered voice assistants and wearables.  
Every **parameter** costs flash/RAM and every **FLOP** burns energy.  
**NdLinear** is a drop-in replacement for `nn.Linear` that factorises weight matrices across tensor dimensions, promising the same (or better) accuracy with far fewer parameters and MACs.  
This repo benchmarks that promise on the **Google Speech-Commands v0.02** dataset.

## 2 • Results

| Model | Rank ratio | Params | FLOPs / inf | Test Acc&nbsp;(3 ep) |
|-------|-----------:|-------:|------------:|---------------------:|
| DS-CNN baseline | n/a | **142 K** | **1 211 M** | 68.7 % |
| NdLinear | 0.50 | **84 K  (-41 %)** | **377 M  (-69 %)** | **71.3 % (+2.6 pp)** |

![image](https://github.com/user-attachments/assets/9b60b962-0447-4ef9-98ae-09debab48605)

![image](https://github.com/user-attachments/assets/668fdc4a-d929-4374-9bc3-82d81a4194e7)


**What does the chart & table show**

* **Left = smaller**, **up = more accurate**.  
  The blue “Nd r = 0.50” point sits **up-and-to-the-left** of the baseline, which means it _dominates_ the baseline in Pareto terms: you get a smaller network **and** a higher accuracy.
* **Parameter drop (-41 %)** By swapping the big fully-connected (FC) layer and the last 1 × 1 convolution for a low-rank NdLinear layer, the total number of trainable weights fell from ≈142 000 to ≈84 000—a 41 % reduction. This results in Smaller file/flash size → easier to store or transmit and Less RAM at inference time.
* **Compute drop (-69 % FLOPs)** Each forward pass now needs 377 million floating-point operations instead of 1 211 million. This results in model running ~3× faster on the same processor.Ccrucial for battery-powered, always-listening devices that perform keyword spotting.
* **Accuracy gain (+2.6 pp after only 3 epochs)** Shows that NdLinear’s “tensor-aware” factorisation doesn’t just compress—it can actually learn a better representation, likely because it preserves the time × frequency structure of the audio spectrogram instead of flattening it.


