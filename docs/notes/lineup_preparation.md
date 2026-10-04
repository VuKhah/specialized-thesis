# Đội hình mô hình 2026-10-03 — việc cần chuẩn bị

Đội hình chốt 2026-10-03 (`Plan.md` mục 4). Note này chỉ liệt kê **việc cần
chuẩn bị** cho từng mô hình; trạng thái làm/chưa làm theo dõi ở `TODO.md`.
⛔ = cần người dùng duyệt (quota GPU, push, việc tay trên UI).

| # | Mô hình | Vai trò | Cách dùng |
|---|---|---|---|
| 1 | Mamba hai chiều B1 (ExtBiMamba chồng) | Trọng tâm | Train từ đầu ~12M |
| 2 | Conformer-12M | Baseline có kiểm soát | Train từ đầu, cùng pipeline |
| 3 | Mamba thuần (một chiều) | Chỉ trích dẫn | Không train — arXiv:2405.12609 |
| 4 | Parakeet-CTC-0.6B-vi | Pre-train ("Conformer có pre-train" thầy nêu) | Zero-shot → fine-tune nếu còn quota |
| 5 | ConExtBiMamba | Ứng viên, quyết sau train thử | Train từ đầu ~12M |
| dự bị | PhoWhisper-small (244M) | Thay #4 nếu Parakeet không chạy được | Zero-shot → fine-tune LoRA |

## 0. Chung cho mọi mô hình (làm trước)

- [x] Push `9d137a7`, `a4fe49e`, `21452f5`, `7fb8492` — 2026-10-04.
- [x] **Bước 3 — code dùng chung** (#1, #2, #5) — code + test CPU 2026-10-04,
      chưa test GPU/DDP (`TODO.md` bước 3): dataset đọc `data/splits/*.tsv`
      + bỏ `excluded.tsv` + `HF_REVISION`; DDP `torchrun` + SyncBN + AMP;
      `DistributedSampler.set_epoch`; `--max_minutes` + checkpoint theo step +
      resume giữa epoch; giải nén tar vào `/tmp/audio_cache`. Thêm tùy chọn
      giới hạn shard (cho train thử 1 shard).
- [ ] ⛔ Việc tay trên UI Kaggle: tạo 5 Dataset từ output 5 kernel
      `make-dataset-asr-*`, chia sẻ sang tài khoản B (hoặc thử gắn
      `kernel_sources` từ tài khoản B trước).
- [x] SpecAugment, CMVN, dropout Mamba — chốt + code 2026-10-04
      (`frontend_decoder_survey.md`).
- [ ] Người dùng xác nhận: `out_proj` chia √28, greedy không LM, chữ số giữ
      nguyên, đo batch 1 fp16.
- [ ] Hiệu đính tay 203 câu clean-test (`corrected_text`, video giữ riêng) — reference cho mọi
      mô hình.
- [ ] Chuẩn hóa văn bản chung trước WER (đã có trong `zero_shot.py`) — dùng cho
      cả mô hình train từ đầu khi chấm clean-test.
- [ ] **Ước lại quota**: số D8 (Conformer 23,2 h, Mamba 16,0 h / 30 epoch) tính
      trên 60.656 câu; sau A1 + giữ riêng video còn 45.442 (×0,75) → khoảng 17,4 h / 12,0 h.
      Kiểm lại bằng số đo thật ở lần chạy thử.

## 1. Mamba hai chiều B1

- [x] Code (`df72f11`), đối chiếu mã gốc + bài ExtBiMamba (`mamba_bidirectional.md`).
- [ ] ⛔ Chạy CUDA thật lần đầu (gộp vào kernel train thử); đo `param_count`
      (kỳ vọng 12.285.696).
- [ ] Train full (tài khoản B) — sau khi chốt kết quả train thử.

## 2. Conformer-12M

- [x] Code; train 2 bước CPU với tokenizer mới.
- [ ] `train.py main()` chưa chạy trên Kaggle — lộ ra ở train thử.
- [ ] Train full (tài khoản A).
- [ ] Ghi rõ trong khóa luận: khối `torchaudio.models.Conformer`, không
      subsampling, không mã hóa vị trí (khác bài Gulati 2020).

## 3. Mamba thuần (chỉ trích dẫn)

- [ ] Lấy số liệu cần trích: Bảng XII (Mamba 40,0 vs ExtBiMamba 37,7 WER test
      LibriSpeech100) + câu "BiMamba consistently outperforms vanilla Mamba".
- [ ] Nêu điều kiện khác (tiếng Anh, ESPnet, có subsampling, baseline độc lập
      thiếu residual — Bảng XVI) để không suy thẳng.
- [ ] Cờ `bidirectional: false` giữ trong code; không lên lịch train.

## 4. Parakeet-CTC-0.6B-vi

- [ ] ⛔ **Kiểm sớm nhất có thể** (quyết định có cần dùng dự bị không): cài NeMo
      trên Kaggle; tìm file `.nemo` (card không có link trực tiếp); chạy trên
      **T4** (card không liệt kê Turing).
- [ ] Zero-shot 203 câu (manifest bản 2 — đã sửa đọc 2026-10-04): rút `ZS_MODELS` của `scripts/kaggle/zero_shot/` về
      `parakeet,phowhisper` (bỏ wav2vec2); bỏ `wer_vi_label`/đọc
      `excluded.tsv` (một mức WER, A1).
- [ ] Chuẩn hóa đầu ra (Parakeet ra hoa/thường + dấu câu) — đã có hàm chung.
- [ ] Đối chiếu dữ liệu pre-train với VietSuperSpeech (card không nêu; mâu
      thuẫn ~2.000 h vs >4.000 h chưa giải) — rủi ro rò rỉ.
- [ ] Fine-tune 600M trên T4: ước VRAM/thời gian bằng vài step (AMP, batch
      nhỏ, có thể đóng băng một phần encoder) trước khi xin quota.
- [ ] RQ2: subsampling 8× → ghi rõ không cùng điều kiện với #1/#2/#5 khi so
      RTF. License: NVIDIA Open Model License — trích trong khóa luận.

## 5. ConExtBiMamba

- [x] Viết `ConExtBiMambaEncoder` (2026-10-04, `src/models/conextbimamba_encoder.py`):
      `½FFN → ExtBiMamba → conv module → ½FFN → LayerNorm`, FFN/conv lấy nguyên
      lớp torchaudio (cùng mã Conformer-12M), cặp Mamba như một lớp B1
      (`reverse_padded`, chung LN, cộng), dropout 0,1, không PE.
- [x] Khớp tham số: 1 lớp ffn 1024 = 2.135.296 → 6 lớp +5,1%, 5 lớp −12% →
      **6 lớp, ffn 928** = **12.241.536 (+0,31%)** (đếm bằng khối giả cùng
      shape mamba-ssm v2.3.1). `configs/model_conextbimamba.yaml`, đăng ký
      `build_encoder`; `param_count.py --others` so cả hai encoder Mamba.
- [x] Test CPU khối giả: tham số, 24 tham số miễn weight decay, câu riêng vs
      trong batch lệch 1,7e-6, nhìn cả quá khứ lẫn tương lai, dropout tắt khi
      eval, SyncBN đổi 6 lớp BN, `train.py` chạy với dữ liệu thật. CUDA thật
      → kernel train thử (thêm vào `CHECK_CODE`).
- [x] Đối chiếu bài (arXiv:2405.12609 v6, Hình 2c, Bảng XII, XIV): macaron ✓,
      Swish ✓ (SiLU trong FFN/conv), dropout ✓, không PE ✓ (Bảng XIV: thêm PE
      không đổi WER). **Lệch có chủ đích, cần người dùng xác nhận:**
      (1) bài giữ kích thước Conformer nên nhiều tham số hơn (34,23M → 41,59M),
      ở đây hạ ffn 1024 → 928 để khớp ~12M; (2) bài khởi tạo A bằng ma trận
      chéo + nhiễu Gauss (tốt nhất trong Bảng XIV), ở đây giữ S4D-Real mặc định
      của mamba-ssm để #1 và #5 dùng cùng khối; (3) không chia `out_proj` (mỗi
      lớp có LN cuối — suy luận của dự án, bài không nói); (4) xoá padding trước
      depthwise conv — Conformer-12M (torchaudio) không xoá.
- [x] Cập nhật `ARCHITECTURE.md` (mục 1, bảng encoder, bản đồ file, 8-e3).

## Train thử (3 mô hình — giữ 2 hay 3)

- Cấu hình chốt: **1 shard, 5 epoch, 3 mô hình** (#1, #5, #2), **1 seed**, ước
  ~2,5-3,5 GPU-h. **Đổi 2026-10-04 (người dùng):** không còn "chọn 1 trong #1/#5";
  sau train thử chốt giữ 2 hay 3 mô hình, mong muốn cả 3 đủ tốt để giữ. Chấp
  nhận 1 seed + biện pháp ổn định (không chạy 2 seed).
- [x] Biện pháp ổn định + đo (2026-10-04): `train.py` bỏ step có loss không hữu
      hạn (đồng bộ mọi rank), đếm step AMP bị GradScaler bỏ, ghi grad norm,
      s/step, VRAM đỉnh, CTC loss + WER val/val_unseen mỗi epoch
      (`metrics.jsonl`), ref/hyp từng câu (`eval_<tập>_epoch<k>.jsonl`);
      `src/evaluation/bootstrap.py` (khoảng tin cậy theo khối video, Liu & Peng
      arXiv:1912.09508). Kernel `scripts/kaggle/train_trial/`: CHECK_CODE CUDA
      thật → chạy ngắn DDP + resume → 3 mô hình tuần tự, trần 90 phút/mô hình,
      mô hình lỗi không chặn mô hình sau → `trial_summary.json`.
- [ ] **Tiêu chí giữ/bỏ — AI đề xuất 2026-10-04 (bản mở rộng), chờ người dùng duyệt
      trước khi chạy.** Bảng so sánh sinh tự động: `src/evaluation/trial_report.py`
      → `trial_report.md` (kernel chạy cuối; chạy lại được ở local trên output).
  **A. Cổng loại cứng** (tự đánh giá; ngưỡng là của dự án, không có tài liệu —
  sửa ở `GATES` trong `trial_report.py`):
  - G0 *lần chạy dùng được:* WER val_unseen của Conformer < 0,95. Trượt → chưa
    kết luận mô hình nào (thêm epoch / so CTC loss val).
  - G1 *không sụp về blank:* câu giải mã rỗng ≤ 5% (val_unseen, epoch cuối).
    CTC dễ kẹt ở đầu ra toàn blank giai đoạn đầu; còn kẹt sau 5 epoch là hỏng.
  - G2 *ổn định:* train loss hữu hạn mọi epoch, step bị bỏ ≤ 1%, WER val giảm
    từ epoch đầu đến cuối. Căn cứ cần cổng này: 2405.12609 mục V báo Mamba chồng
    khó train ổn định.
  - G3 *tài nguyên:* ước train full 30 epoch ≤ 30 GPU-giờ, VRAM đỉnh ≤ 14 GiB.
  **K. Tiêu chí cân nhắc** (báo số, người dùng/GVHD quyết):
  - K1 *chất lượng so với mốc:* hiệu WER so với Conformer-12M trên val và
    val_unseen, CI95 bootstrap theo khối video, P(tốt hơn mốc).
  - K2 *xu hướng học:* đường WER theo epoch, mức giảm WER tương đối ở epoch
    cuối — còn giảm mạnh thì thứ hạng sau train full dễ đổi.
  - K3 *tổng quát hoá:* khoảng cách WER val_unseen − val (video chưa gặp vs đã gặp).
  - K4 *dạng lỗi:* tỉ lệ thay/xoá/chèn, CER — xoá nhiều bất thường là dấu hiệu
    học chưa tới (gần sụp blank).
  - K5 *đánh đổi:* tham số, s/step, VRAM, ước GPU-giờ, RTF eval (proxy) và vị trí
    Pareto theo (WER val_unseen, GPU-giờ). Đề tài hỏi về đánh đổi WER – tốc độ –
    tài nguyên → mô hình bị trội cả hai trục đóng góp ít điểm mới cho bảng đánh đổi.
  **C. Quy tắc đề xuất:** qua A thì mặc định giữ. Bỏ một mô hình qua A chỉ khi
  vừa kém mốc có ý nghĩa (CI95 hiệu WER val_unseen > 0) vừa bị trội Pareto, và
  người dùng đồng ý. Trượt A → bỏ, hoặc sửa rồi train thử lại nếu là lỗi code.
  Lưu ý cho người dùng: mô hình #1 B1 (Mamba thuần) là trọng tâm đề tài — bỏ B1 là đổi đề
  tài, nên bàn với GVHD dù số liệu có ra sao.
  - Ngân sách nếu giữ 3: train full ~18,5 (Conformer) + ~12,8 (B1) + ? (ConExt,
    đo ở train thử) GPU-h — xếp lại lịch hai tài khoản (Plan mục 2).
- [ ] Ghi rõ hạn chế: 1 seed, 5 epoch trên 1/4 dữ liệu — thứ hạng sớm có thể
      đảo (successive halving/Hyperband chấp nhận rủi ro này), dùng để phát hiện
      mô hình bất ổn/tệ rõ, không để xếp hạng cuối.

## Dự bị — PhoWhisper-small

- **Khi nào dùng:** Parakeet không cài/chạy được trên T4, hoặc kết quả
  zero-shot bất thường không giải thích được. Người dùng quyết chuyển.
- [ ] Zero-shot chạy kèm Parakeet ở cùng kernel (rẻ, ~vài phút cho 203 câu)
      — có số sẵn nếu phải chuyển.
- [ ] Fine-tune LoRA (`peft`) trên T4: ước VRAM/thời gian bằng vài step.
- [ ] Encoder-decoder, cửa sổ 30 s: đoạn 10-15 s không sao; **RQ2 audio
      ghép dài phải cắt khúc** → ghi rõ khi so RTF.
- [ ] Dữ liệu fine-tune 844 h có phần "dữ liệu riêng" — không kiểm được rò rỉ
      với VietSuperSpeech, nêu hạn chế. License BSD-3-Clause.

## RQ2 (đo cho mọi mô hình)

- [ ] Script ghép đoạn liên tiếp cùng video (train + val, bỏ `excluded.tsv`);
      AI đề xuất các mức độ dài, người dùng duyệt.
- [ ] Sửa `rtf.py` bỏ bucket 3-30 s; đo RTF/độ trễ/VRAM; điểm giao
      Mamba vs Conformer bằng trọng số ngẫu nhiên (đo sớm được).
