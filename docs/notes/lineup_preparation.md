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
- [ ] Hiệu đính tay 192 câu clean-test (`corrected_text`) — reference cho mọi
      mô hình.
- [ ] Chuẩn hóa văn bản chung trước WER (đã có trong `zero_shot.py`) — dùng cho
      cả mô hình train từ đầu khi chấm clean-test.
- [ ] **Ước lại quota**: số D8 (Conformer 23,2 h, Mamba 16,0 h / 30 epoch) tính
      trên 60.656 câu; sau A1 còn 48.340 (×0,80) → khoảng 18,5 h / 12,8 h.
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
- [ ] Zero-shot 192 câu: rút `ZS_MODELS` của `scripts/kaggle/zero_shot/` về
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

- [ ] Viết `ConExtBiMambaEncoder`: khối `½FFN → ExtBiMamba → conv module →
      ½FFN → LayerNorm` (theo 2405.12609; torchaudio không cho thay attention
      → khối tùy biến), dùng lại `reverse_padded` + cặp khối Mamba của B1.
- [ ] Khớp ~12M (ước ~2,1M/khối → ~6 khối; tính lại chính xác), thêm
      `configs/model_conextbimamba.yaml`, đăng ký trong `build_encoder`.
- [ ] Test CPU bằng khối Mamba giả (như B1); CUDA thật ở train thử.
- [ ] Đối chiếu chi tiết khối với bài (Bảng XIV: macaron, Swish, dropout,
      không positional encoding) trước khi chạy.
- [ ] Cập nhật `ARCHITECTURE.md` (encoder thứ ba).

## Train thử (chọn giữa #1 và #5)

- Cấu hình chốt: **1 shard, ~5 epoch, 3 mô hình** (#1, #5, #2 làm mốc),
  ước ~2-3 GPU-h; gộp benchmark lần 3.
- [ ] AI đề xuất **tiêu chí chọn** (ví dụ: WER val ở epoch cuối, độ dốc
      đường loss, có bất ổn/NaN không, s/step, VRAM), người dùng duyệt
      **trước khi** chạy.
- [ ] Ghi rõ hạn chế: thứ hạng sớm chỉ để loại phương án tệ rõ.

## Dự bị — PhoWhisper-small

- **Khi nào dùng:** Parakeet không cài/chạy được trên T4, hoặc kết quả
  zero-shot bất thường không giải thích được. Người dùng quyết chuyển.
- [ ] Zero-shot chạy kèm Parakeet ở cùng kernel (rẻ, ~vài phút cho 192 câu)
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
