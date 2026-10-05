# Chính sách số học khối Mamba — tra tài liệu (2026-10-05)

Bối cảnh: B1 ra NaN dưới autocast fp16 (T4 không có bf16); kernel `diag-b1-asr`
cho thấy tràn ở **trung gian bên trong khối Mamba** (Inf đầu tiên tầng 8 nhánh
xuôi, đầu ra khối fp32 max ~7e3). Đang dùng `mamba_fp32` (cả khối fp32, mọi
encoder có Mamba) — Plan.md nhật ký 2026-10-05 (tối). Người dùng yêu cầu tra thêm
trước khi chốt. **Trạng thái: đề xuất, chờ người dùng chốt.**

## 1. Bằng chứng

| Nguồn | Điều kiện | Nói gì / làm gì | Áp vào dự án |
|---|---|---|---|
| README mamba-ssm | LLM | "SSMs are sensitive to their recurrent dynamics"; nếu bất ổn, thử AMP (tham số fp32) trước | Dự án đã AMP tham số fp32 — vẫn NaN vì tràn ở **kích hoạt**, không ở tham số |
| U-Mamba (Ma và cs., 2024), code `MambaLayer` | phân đoạn ảnh y tế, nnU-Net AMP fp16 | `@autocast(enabled=False)` + ép fp16/bf16 → fp32 trước khối Mamba | **Đúng cách `mamba_fp32` đang làm** — tiền lệ có code công bố |
| ConMamba (Jiang và cs., 2024; `xi-j/Mamba-ASR`) | ASR, LibriSpeech | `precision: bf16` | bf16 cần GPU sm_80+, T4 (sm_75) không có → không áp được |
| Miyazaki và cs., Interspeech 2024 (arXiv:2406.16808), ESPnet | ASR 7 tập, A100 | "Mamba … sometimes unstable during training and tended to overfit" → dropout 0,2 sau in/out proj + AdamW; không nói precision | Dự án đã AdamW + dropout 0,1; không nhắm tràn số |
| Zhang và cs. (arXiv:2405.12609), ESPnet | ASR | "Mamba model was sometimes unstable"; clip gradient [−1, 1]; không nói precision | Không nhắm tràn forward |
| Jamba (Lieber và cs., arXiv:2403.19887) | LLM lai, ổn ≤ 1,3B, spike ở 7B | "inner parts of the Mamba layers suffer from large activation values" → thêm RMSNorm vào kích hoạt trong | Cùng **hiện tượng** (kích hoạt trong khối lớn dần) nhưng khác quy mô; là đổi kiến trúc |
| FalconMamba 7B (arXiv:2410.05355) | LLM 7B | loss spike → RMSNorm sau B, C, Δ | Đổi kiến trúc; đường `mamba_inner_fn` hợp nhất không lộ B, C, Δ → phải dùng đường chậm |
| Halloran và cs., TMLR (arXiv:2406.00209) | **fine-tune** LLM đã pre-train | Mamba ổn định dưới mixed-precision fine-tuning | Chỉ fine-tune, không phải train từ đầu → không mâu thuẫn quan sát của dự án |

Không tìm được bài ASR nào báo train Mamba bằng fp16 thuần thành công **hoặc** so
sánh có kiểm soát fp16/fp32 cho khối Mamba.

## 2. Phương án

| | Cách | Bằng chứng | Giữ kiến trúc? | Chi phí |
|---|---|---|---|---|
| **P1 (đang dùng)** | Cả khối Mamba fp32 (`mamba_fp32`) | U-Mamba (code), README mamba-ssm (tinh thần) | Có | B1 0,733 vs 0,47 s/step (+55%), ước 18,1 GPU-h/30 epoch; ConExt cần đo lại |
| P2 | RMSNorm trong khối (Jamba/FalconMamba) + giữ fp16 | Chỉ LLM 7B; không có ở ASR | **Không** — B1 không còn là Mamba gốc, phải viết lại khối, train thử lại | Code mới + rủi ro |
| P3 | bf16 (ConMamba) | ConMamba | Có | Không chạy được trên T4 |
| P4 | Giữ fp16, tăng dropout/giảm LR/clip (ESPnet) | Không nhắm tràn số | Có | Không có gì bảo đảm hết NaN; quan sát diag: kích hoạt tăng theo train |

## 3. Đề xuất

Giữ **P1** cho cả B1 và ConExtBiMamba (đồng nhất số học trong nhóm Mamba), áp cả
lúc đo RQ2 (mô hình train ở fp32 khối Mamba thì suy luận cũng vậy; pilot RQ2 đã đo
theo cách này). Khi viết: nêu rõ là điều kiện phần cứng (T4 không bf16), chi phí
+55% s/step của B1 là chi phí của lựa chọn này, không phải bản chất Mamba — ghi
kèm s/step fp16 đo được trước khi NaN (0,47) để người đọc thấy biên.

## Nguồn

- mamba-ssm README: https://github.com/state-spaces/mamba
- U-Mamba `UMambaEnc_3d.py`: https://github.com/bowang-lab/U-Mamba/blob/main/umamba/nnunetv2/nets/UMambaEnc_3d.py
- ConMamba yaml: https://github.com/xi-j/Mamba-ASR/blob/main/hparams/CTC/conmamba_large.yaml
- Miyazaki và cs., Exploring the Capability of Mamba in Speech Applications: https://arxiv.org/abs/2406.16808
- Zhang và cs., Mamba in Speech: https://arxiv.org/abs/2405.12609
- Jamba: https://arxiv.org/abs/2403.19887
- Falcon Mamba: https://arxiv.org/abs/2410.05355
- Halloran và cs., Mamba SSMs Are Lyapunov-Stable Learners: https://arxiv.org/abs/2406.00209
