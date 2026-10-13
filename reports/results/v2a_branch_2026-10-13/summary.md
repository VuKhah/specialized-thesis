# Nhóm rẽ nhánh v2a — tóm tắt (sinh bởi scripts/plot_v2a_branch.py)

WER trên nhãn giả, CTC greedy, 1 seed; chưa có CI. Checkpoint bảng chính = ep37 (chốt trước khi có số).

| Mô hình | Tập | v1 ep29 (điểm rẽ) | v1 best (ep) | v1 avg ep20–29 | **v2a ep37** | ep37 − v1 best |
|---|---|---|---|---|---|---|
| Mamba B1 | val | 0.1587 | 0.1562 (ep28) | 0.1388 | **0.1271** | -2.91 điểm |
| Mamba B1 | val_unseen | 0.1563 | 0.1547 (ep28) | 0.1361 | **0.1252** | -2.95 điểm |
| ConExtBiMamba | val | 0.1900 | 0.1875 (ep28) | 0.1671 | **0.1525** | -3.50 điểm |
| ConExtBiMamba | val_unseen | 0.1876 | 0.1846 (ep28) | 0.1635 | **0.1506** | -3.40 điểm |
| Conformer-12M (không PE) | val | 0.2732 | 0.2519 (ep28) | 0.2285 | **0.2094** | -4.25 điểm |
| Conformer-12M (không PE) | val_unseen | 0.2681 | 0.2481 (ep28) | 0.2250 | **0.2068** | -4.13 điểm |

clean_test (203 câu, mới hiệu đính 11 câu) chỉ có cho v1: Mamba B1 best 0.1711 / avg 0.1556; ConExtBiMamba best 0.2028 / avg 0.1833; Conformer-12M (không PE) best 0.2646 / avg 0.2392. v2a ep37 chưa chấm clean_test.

## Chi phí train đo được (2×T4, giờ đồng hồ train + eval, không gồm setup kernel)

| Mô hình | s/step v1 (trung vị) | s/step pha giảm (trung vị) | Giờ v1 (30 ep) | Giờ pha giảm (8 ep) | VRAM đỉnh | Step loss không hữu hạn |
|---|---|---|---|---|---|---|
| Mamba B1 | 0.682 | 0.688 | 16.67 | 4.55 | 3.61 GiB | 0 |
| ConExtBiMamba | 0.397 | 0.399 | 9.97 | 2.70 | 3.18 GiB | 0 |
| Conformer-12M (không PE) | 0.260 | 0.260 | 6.58 | 1.80 | 2.88 GiB | 0 |
