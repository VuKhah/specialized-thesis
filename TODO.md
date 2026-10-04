# TODO tổng — Mamba vs Conformer ASR

Cập nhật lần cuối: **2026-10-04** (bước 3 code dùng chung; chốt front-end CMVN + SpecAugment, dropout Mamba, giữ CTC). **Nguồn sự thật duy nhất cho việc cần làm /
đang chặn / đã xong.** Kế hoạch tuần + quyết định + nhật ký phiên ở
[`Plan.md`](Plan.md); kiến trúc ở [`ARCHITECTURE.md`](ARCHITECTURE.md).

Câu hỏi cần mang đi hỏi (người dùng/GVHD): [`QA.md`](QA.md).

## 🔴 Đang chặn — chờ quyết định từ GVHD

- [ ] **ĐỔI ĐỊNH HƯỚNG sau buổi GVHD lần 1 (ghi 2026-10-01) — ưu tiên số 1**:
      khảo sát Mamba (trọng tâm) vs Conformer pre-train + fine-tune vs 2-3 mô
      hình khác vs 1 baseline; không cần giống hệt cấu hình; nhận xét theo đánh
      đổi WER/tốc độ/tài nguyên. Cần làm rõ Q-a…Q-g trong
      `docs/notes/gvhd_buoi_1.md` mục 4 (danh sách mô hình, Mamba train từ đầu
      hay fine-tune, baseline, fine-tune hay zero-shot, đề cương/tên đề tài).
      **Chờ chốt trước bước 6 (train) và trước khi sửa phần riêng Conformer.**
      Bước 5 (tạo dataset) và hiệu đính clean-test vẫn làm được.
      Đã biết thêm: thầy không gợi ý mô hình, dặn **"đừng ôm đồm"**; **sẽ sửa
      đề cương** (làm bản mới, không sửa docx cũ); buổi sau dự kiến Tuần 9.
      A1 (nhãn tiếng Anh) thầy giao **người dùng tự quyết** sau khi tìm hiểu
      ngôn ngữ pre-train của từng mô hình. A2-A4: chưa được trả lời.
      **AI đã tra ứng viên + phân tích A1:** `docs/notes/survey_model_candidates.md`
      (đề xuất 4 mô hình: Mamba + Conformer nhỏ train từ đầu, Parakeet-CTC-0.6B-vi,
      PhoWhisper-small; A1 nghiêng lọc khỏi train + báo clean-test 2 mức).
      **Chốt 2026-10-02: đủ 5 mô hình** (Mamba hai chiều + Conformer nhỏ train
      từ đầu, Parakeet-CTC-0.6B-vi, PhoWhisper-small, wav2vec2-base-vi); **duyệt
      mọi thay đổi phép đo** (RQ2 = so hiệu quả giữa mô hình + ghép đoạn đo RTF
      theo độ dài; chuẩn hóa văn bản chung trước WER; A3 giữ + nêu hạn chế,
      tách nhóm zero-shot/fine-tune khi bàn). **A1 chốt 2026-10-03** (mục dưới). Còn: viết đề cương điều chỉnh
      (bản mới) trước buổi Tuần 9.

- [x] **RQ2 — CHỐT 2026-10-03 (người dùng): phương án 2 + đổi cách hỏi.**
      RQ2 thành so hiệu quả giữa các mô hình (RTF, độ trễ, VRAM, tham số,
      GPU-giờ); dải độ dài lấy từ **audio ghép nhân tạo** (đoạn liên tiếp cùng
      video, chỉ để đo, không train); WER chỉ trên đoạn gốc 10-15 s. Việc code
      ở 🟡 "RQ2". Ghi chú gốc bên dưới.
      Sai lệch số liệu: số liệu thật
      (67.405 mẫu/245,42h, audio gần như đồng nhất 10-15s) khác đề cương đã
      đăng ký (52.023/267,39h, dải 3-30s) — RQ2 (RTF theo độ dài audio) không
      còn đủ dải để chứng minh ưu thế O(n) của Mamba. 4 phương án trong
      `docs/notes/dataset_discrepancy.md`.
      *Không chặn train (rà 2026-09-30):* cả 4 phương án chỉ đổi phần đo RTF
      sau train (RTF không phụ thuộc trọng số → đo được song song lúc train).
      Chỉ ảnh hưởng train nếu GVHD muốn WER trên audio dài hoặc thêm dữ liệu
      train — nên hỏi thêm: "RQ2 chỉ cần RTF hay cần cả WER audio dài?"
      *AI rà 2026-10-03:* các đoạn số seg liền nhau **không liền về thời gian**
      (87% đoạn < 15 s kết thúc ở chỗ lặng đã bị cắt bỏ; 13% cắt cứng ở 15,0 s;
      5,9% số seg khuyết). Ghép vẫn hợp lệ để đo RTF/độ trễ/VRAM theo độ dài
      (chi phí chỉ phụ thuộc số khung), gọi là "audio ghép nhân tạo". **Không**
      đo được WER audio dài: validation chỉ là ~10% đoạn rải từ chính video
      train → chuỗi seg liên tiếp ≥ 60 s chỉ có 1. Đề xuất: RQ2 chỉ đo hiệu quả
      theo độ dài, ghép từ cả train + val; WER chỉ trên đoạn 10-15 s.

- [x] **A1 — nhãn sai do audio không phải tiếng Việt — CHỐT 2026-10-03
      (người dùng): lọc C ∩ L.** Người dùng nghe kiểm: 149 video phỏng vấn khách
      nước ngoài (tiếng Anh) + video Alive Kicking Ep4 (tiếng Nhật, nhãn trông
      như tiếng Việt → lọc theo nhãn không đủ). LID Whisper-small trên audio
      (kernel `lid-asr`, ~65 phút T4) khớp các nhóm đã nghe. Loại nếu LID ≠ `vi`
      HOẶC thuộc 149 video HOẶC nhãn < 20% dấu → `data/splits/excluded.tsv`
      (`python -m src.data.filter_language`). Train còn **48.340 câu/175,77 h**
      (từ 220,84 h), val 5.140/18,65 h, **clean-test 192 câu**. Khóa luận: coi
      dữ liệu là tiếng Việt, một mức WER trên 192 câu, không bàn tiếng Anh.
      Chi tiết: `docs/notes/survey_model_candidates.md` mục 6.

- [x] **Mamba hai chiều — CHỐT 2026-10-02 (người dùng), phương án B1**; việc
      sửa code chuyển xuống 🟡. Ghi chú gốc: Mamba hiện chỉ nhìn quá khứ, Conformer nhìn cả câu →
      biến gây nhiễu cho RQ1. Phương án B1 (hai chiều, 14 lớp × 2 khối, cộng) =
      12.285.696 tham số (+0,67%), vẫn 28 khối như cũ; cần đảo chuỗi theo độ dài
      thật, benchmark lại. Chi tiết: `docs/notes/mamba_bidirectional.md`. **Chốt
      trước khi train; hỏi GVHD (A4).** Code B1 xong (`df72f11`), chưa chạy CUDA thật.

## 🟡 Sẵn sàng làm ngay — kế hoạch Kaggle đã duyệt 2026-09-28

Chi tiết + lý do: `docs/notes/training_plan_kaggle.md` mục 5 (D1-D11 đã
duyệt, D3 → **4 shard**, train full + đánh giá theo epoch).

- [x] **1. Kernel CPU (D9) — xong 2026-09-28** (`scripts/kaggle/check_env/`):
      `/tmp` trống ~1.200 GB, tải HF ẩn danh ~2,4 h cho 67.405 file (0 lỗi),
      output giữ đủ 600 file, tar 590 MB/s. Chi tiết: `training_plan_kaggle.md` mục 5.
- [x] **2. Kernel GPU benchmark (D9) — xong 2026-09-30** (~16 phút quota):
      bảng đủ 8 cấu hình ở `training_plan_kaggle.md` mục 5. Tóm tắt: workers
      không giúp; AMP Conformer 0,56 s/step (17,7 h/30 epoch), Mamba 1,02
      (32,1 h); DDP+AMP Conformer **chậm đi** 0,77 (24,4 h), Mamba 0,48 (15,0 h).
- [x] **D8 chốt 2026-09-30: cả hai encoder DDP 2 GPU + SyncBN + AMP**, batch
      toàn cục 16 (8/GPU). Ước 30 epoch: Conformer 23,2 h, Mamba 16,0 h (chưa
      tính eval). Số đo: `training_plan_kaggle.md` mục 5.
- [x] **`PACK_TAR = True` chốt 2026-09-30**: 5 dataset đều là tar; đầu mỗi phiên
      train giải nén vào `/tmp/audio_cache` (**không** `/kaggle/working`).
- [x] **3. Sửa code dùng chung (D5, D7, D8, D10) — code xong 2026-10-04,
      chưa commit.** `manifest_indices` đọc `data/splits/*.tsv` + bỏ
      `excluded.tsv` (train 48.340, `train_shard0` 12.086, val 5.140,
      clean-test 192); `HF_REVISION`; `AUDIO_CACHE_DIR` từ biến môi trường;
      `train.py`: `torchrun` DDP + SyncBN + AMP (front-end + CTC loss fp32),
      sampler tất định theo (seed, epoch) bỏ được step đã học, `--max_minutes`,
      checkpoint mỗi 20 phút ghi nguyên tử, `--resume_from`, `--train_manifests`,
      `--epochs`, eval chia GPU; `src/data/extract_audio.py` giải nén tar. Lịch LR
      giữ warmup + hằng số.
      **Đã test CPU local (Conformer):** resume giữa epoch ra trọng số **giống hệt**
      chạy liền (max |dW| = 0, tắt dropout/augment); `eval_clean_test` 192 câu;
      giải nén tar thật. **Chưa test được:** DDP (PyTorch Windows không có gloo
      device), NCCL/SyncBN/AMP, Mamba → bước đầu của kernel train thử.
      Khởi động: `load_dataset` ~45 s/lần kể cả đã cache → làm ấm bằng
      `python -m src.data.vietsuperspeech_dataset` rồi chạy `HF_HUB_OFFLINE=1`
      (0,2 s).
- [x] **Front-end + dropout — chốt và code 2026-10-04** (người dùng duyệt, căn
      cứ `docs/notes/frontend_decoder_survey.md`): giữ CTC; CMVN toàn cục
      (`configs/cmvn_stats.json`, 2000 câu train seed 42); SpecAugment 2 freq
      mask F=27 + 10 time mask ≤ 5%, không time warp, chỉ lúc train; dropout 0,1
      cho Mamba (đầu ra mỗi khối); giữ log-mel 80/25/10 ms. Mục `features` trong
      2 yaml, dựng model qua `train.build_model`. Test: CMVN đúng công thức, mask
      không lọt padding (~22% khung, ~33% kênh), dropout tắt khi eval, bất biến
      padding Mamba giữ (khối giả, 9,5e-7), train 3 epoch CPU loss 113 → 41.
- [x] **Theo khuyến nghị tác giả Mamba — sửa 2026-09-30** (người dùng duyệt):
      `build_optimizer` miễn weight decay cho `A_log`/`D`, `weight_decay: 0.01`
      ghi rõ trong 2 yaml; `MambaEncoder` thêm `norm_f`, chia `out_proj` cho
      √n_layers, residual fp32. Test local: Conformer giống hệt từng bit so với
      trước; Mamba kiểm bằng khối giả (không CUDA). **Còn: chạy khối Mamba thật
      trên Kaggle** (gộp vào lần chạy GPU kế tiếp) + đo lại `param_count`
      (dự kiến 12.292.864, +0,73%).
- [x] **4. Manifest 4 shard + val — xong 2026-09-30** (`src/data/make_shards.py`
      → `data/splits/`, ~9 MB TSV): kiểm rời nhau, hợp = 60.656, không lẫn
      validation, val 6.499 + clean-test 250 (đối chiếu nội dung theo index).
      Mỗi shard 15.164 câu / 55,21 h / 6,36 GB, lệch phân bố ≤ 0,1 điểm %, phủ
      996/998 token BPE. Đã commit + push 2026-09-30.
- [ ] **5. Kernel CPU tạo 5 dataset** — *2026-10-02: 5 kernel
      `make-dataset-asr-{shard0..3,val}` **xong cả 5, qua kiểm tra**
      (15.164×4 + 6.499 file, 0 lỗi, 16 kHz mono, duration ≤ 0,1 s; ~6 GiB/shard,
      val 2,6 GiB). **Còn (việc tay trên UI):** tab Output của từng kernel →
      New Dataset, private, tên đề xuất `vss-asr-train-shard{0..3}`,
      `vss-asr-val`; **không push version mới** các kernel này trước khi tạo
      xong; chia sẻ sang tài khoản B; ghi slug dataset vào đây.*
      (4 train + 1 val), private —
      `scripts/kaggle/make_dataset/make_dataset.py` (`PACK_TAR = True`), test
      local 20 file thật. Còn: chạy 5 lần (sửa `PART`), mỗi lần ~40 phút CPU
      (không tốn GPU); tạo Dataset từ output; chia sẻ sang tài khoản B (D6).
- [ ] **6. Train full song song**: tài khoản A Conformer, B Mamba; báo cáo
      theo epoch, so hai mô hình ở cùng số epoch.

## 🟡 Mới 2026-10-02 — chờ duyệt chạy

- [x] **Commit + push** `df72f11` (Mamba B1, chuẩn hóa văn bản, kernel
      zero_shot, notes) — push 2026-10-03.
- [x] **Commit + push** thay đổi 2026-10-03 + 2026-10-04 (`9d137a7`, `a4fe49e`,
      `21452f5` code bước 3 + front-end, `7fb8492` Plan) — push 2026-10-04.
- [x] **Train lại tokenizer BPE trên 48.340 nhãn đã lọc — xong 2026-10-03**
      (người dùng duyệt; `python -m src.data.filter_language --train-tokenizer`).
      Cùng tham số, vocab vẫn 1001 (gồm blank) → số tham số hai mô hình không
      đổi. 121/1000 piece đổi: bỏ mảnh giả tiếng Anh (`AND`, `ATION`, `KING`…),
      thêm âm tiết Việt; piece có dấu 663 → 771. Val sạch: 59,7 token/câu (cũ
      60,7), 0 unk, decode khớp 5.332/5.332. Conformer: 2 bước train thật CPU,
      loss 136,7 → 80,1. Mamba chưa test (cần CUDA; tokenizer không chạm
      encoder). `data/splits/summary.json` (`bpe_tokens`) còn tính theo
      tokenizer cũ — chạy lại thống kê khi sửa `make_shards`.
- [ ] Kernel zero_shot: sau A1 chỉ cần **một mức WER trên 192 câu** (bỏ
      `wer_vi_label` hoặc đọc `excluded.tsv`) — sửa trước khi chạy.
- [ ] **Kernel GPU benchmark lần 3** (Mamba B1, `CHECK_CODE` kiểm CUDA thật;
      ~10-15 phút) — xin duyệt quota.
- [ ] **Kernel GPU zero_shot** 3 mô hình trên 250 câu clean-test (~0,5 GPU-h)
      — xin duyệt quota. Kiểm Parakeet chạy được trên T4.
- [ ] **RQ2 (chốt 2026-10-03, chưa code):** script ghép đoạn liên tiếp cùng
      video (từ train + val, đã bỏ `excluded.tsv`) thành audio dài theo các mức
      độ dài; sửa `rtf.py` bỏ bucket 3-30 s, đo RTF/độ trễ/VRAM cho cả 5 mô
      hình. Không phụ thuộc trọng số → chạy được với trọng số ngẫu nhiên
      trước khi train xong. Mức độ dài cụ thể: AI đề xuất, người dùng duyệt.
- [ ] **Đội hình mới 2026-10-03 + train thử BiMamba vs ConExtBiMamba** (người
      dùng chốt hướng; Plan mục 4). Việc: viết `ConExtBiMambaEncoder` (khối
      Conformer tùy biến, ½FFN → ExtBiMamba → conv → ½FFN → LN; dùng lại
      `reverse_padded`/khối B1), khớp ~12M; train thử trên tập nhỏ vài giờ
      GPU. **Chốt 2026-10-03:** baseline = **Conformer-12M** train từ đầu;
      pre-train = **Parakeet-CTC-0.6B-vi**, **dự bị PhoWhisper-small** (bỏ
      wav2vec2 → kernel zero_shot rút về 2 mô hình; vẫn phải kiểm NeMo/T4); train thử = **1 shard,
      ~5 epoch, 3 mô hình** (BiMamba, ConExtBiMamba, Conformer làm mốc), ước
      ~2-3 GPU-h, so WER val + đường loss + s/step + VRAM. Thứ hạng sớm chỉ để
      loại phương án tệ rõ. Còn: tiêu chí chọn cụ thể (AI đề xuất, người dùng
      duyệt). Thứ tự: bước 3 → `ConExtBiMamba` + test CPU khối giả → kernel
      train thử (gộp benchmark lần 3).
      **Danh sách chuẩn bị từng mô hình: `docs/notes/lineup_preparation.md`.**
- [x] **Mamba chồng thuần hay thêm FFN — CHỐT 2026-10-04 (người dùng): giữ B1,
      không làm biến thể B1 + FFN**; ConExtBiMamba (đã trong train thử) là
      phương án "có FFN". Căn cứ 2405.12609 Bảng XVI (+FFN chỉ ~1,6 điểm) và XII
      (khung Conformer mới cải thiện lớn) — `docs/notes/frontend_decoder_survey.md` mục 5.
- [ ] Xác nhận các thiết lập tác nhân tự chọn: `out_proj` chia √28; greedy
      không LM; chữ số giữ nguyên; đo batch 1 fp16.
- [x] Viết lại `Plan.md` mục 1-5, `CLAUDE.md`, `README.md` theo định hướng
      khảo sát — xong 2026-10-04.
- [ ] Khung **đề cương mới** (dựa `Plan.md` mục 1-2) trước buổi GVHD Tuần 9.

## 🟢 Việc tay — song song bất cứ lúc nào, không chặn code

- [x] ~~Khi hiệu đính clean-test: thêm cột nhãn ngôn ngữ audio~~ — thay bằng
      LID trên audio + người dùng nghe mẫu (A1 chốt 2026-10-03).

- [ ] Nghe & hiệu đính **192 câu còn lại sau A1** (bỏ 58 câu trong
      `excluded.tsv`) trong `data/processed/clean_test_manifest.json`
      (cột `corrected_text` còn trống) — dùng mục 5 của
      `notebooks/02_dataset_eda.ipynb`.

## 📘 Khóa luận (bản viết) — kế hoạch, mốc và quy cách ở `Plan.md` mục 6

**Làm ngay / sớm**
- [ ] **Gặp GVHD (tối đa 2 tuần/lần, GVHD ký logbook)**: cho AI biết ngày gặp
      gần nhất + ngày hẹn kế tiếp để ghi vào bảng ở `Plan.md` mục 6. Nên hỏi:
      🔴 quyết định RQ2 · hình thức nộp hiện hành (biểu mẫu 2018) · dạng "thiết
      kế" ở Chương 2. *(việc tay của người dùng)*
- [ ] **Sao lưu file Word khóa luận lên Drive** `specialized-thesis/khoa_luan/`
      (`docs/khoa_luan/KhoaLuan_Mamba_vs_Conformer_ASR.docx`, gitignore). Đây là
      **bản làm việc duy nhất**; script sinh file đã bỏ — đừng dựng lại kẻo mất
      bản sửa tay.
- [ ] **Điền chỗ trống trên bìa** (tô vàng): Bộ môn, Khóa, logo khoa. Xác nhận
      tên đề tài trên bìa: bìa ghi `Conformer`, đề cương đã nộp ghi
      `Con-Former` — hỏi GVHD nên theo bản nào.
- [ ] **Rà bản nháp Chương 1** (~9 trang, đoạn tô vàng = ghi chú cần xử lý):
      đối chiếu 12 trích dẫn với nguồn gốc (điền từ trí nhớ AI, chưa kiểm),
      đọc lại 3 bài Mamba-ASR (số liệu AISHELL-1 lấy từ đề cương), chuyển công
      thức sang Word Equation, vẽ Hình 1.1-1.3. Mục tiêu xong trước Tuần 7.
- [ ] **Bắt đầu ghi Tài liệu tham khảo** ngay khi trích dẫn (chỉ tài liệu thực
      sự trích dẫn; quy ước ghi ở Mẫu 5).

**Theo tuần (chưa đến hạn)**
- [ ] Tuần 6-7: Ch2 Phân tích & thiết kế (từ `ARCHITECTURE.md`).
- [ ] Sau Tuần 8/12/13: Ch3 Thực nghiệm — chèn kết quả từ `reports/`; phần mô
      tả dữ liệu chờ 🔴.
- [ ] Tuần 14: Ch4 Demo & tổng kết. Tuần 15: Mở đầu, Kết luận, Tóm tắt, Phụ lục.
- [ ] **Trước hết đợt Kaggle:** sao lưu `best.pt` của cả hai thí nghiệm lên
      Drive (cần cho đĩa CD, mục `SETUP`).
- [ ] Khi nộp: danh sách kiểm tra ở `Plan.md` mục 6 (2 cuốn + 2 CD, đề cương có
      chữ ký, GVHD duyệt trước khi in).

## ⚠️ Cần kiểm tra / rủi ro đã biết (không phải việc làm ngay)

**Repo public**
- [ ] `notebooks/02_dataset_eda.ipynb` nhúng audio YouTube (20 output
      `<audio>`, ~5MB) — trong repo public và lịch sử git. Người dùng chọn bỏ
      qua (2026-09-19). Chưa kiểm license VietSuperSpeech. Khi xử lý: xoá
      output (`nbstripout`); muốn gỡ khỏi lịch sử phải viết lại lịch sử + force
      push (hỏi trước).
- [ ] Lịch sử git vẫn chứa các file đã gỡ index: `.docx` (đề cương, biểu mẫu
      khoa) và file TLCN có tên+MSSV người thứ ba trong tên file. Chưa quyết
      định có viết lại lịch sử không.
- [ ] `docs/de_cuong/De_Cuong_Chi_Tiet_Mamba_ASR.docx` trên đĩa **khác bản đã
      commit** (trước đó `git status` báo modified; nghi Word tự lưu). Giờ file
      không còn track nên git không hiện diff nữa — mở kiểm tra nội dung trước
      khi upload Drive, đừng để mất bản edit tay. Không sửa nội dung file.

**Code / thiết kế** (chi tiết ở `ARCHITECTURE.md` mục 8)
- [ ] `train.py` viết lại 2026-10-04 **chưa chạy trên Kaggle** (DDP/NCCL,
      SyncBN, AMP, Mamba, `extract_audio`) — kernel train thử phải chạy một
      vòng ngắn (vài step + cắt `--max_minutes` + resume) trước khi train thật.
- [ ] **Ước lượng thời gian train vượt hạn mức**: ~1,3 s/step (batch 16, T4,
      chưa tính tải dữ liệu) × 3.791 step/epoch ≈ 80 phút/epoch → 30 epoch ≈
      40 GPU-giờ **mỗi mô hình**, trong khi hạn mức Kaggle 30 GPU-giờ/tuần.
      Liên quan mục `train.py` không AMP/1 GPU bên dưới — chưa quyết.
      *2026-09-30: đã đo (fp32 1 GPU: Conformer 42,5 h, Mamba 48,7 h); lựa
      chọn tăng tốc là mục ⏸ D8 ở trên.*
- [ ] Mamba đơn hướng (đang cân nhắc hai chiều — 🔴 ở trên), chưa mask padding; hai
      encoder không subsampling (T'=T ≈ 1000-1500 khung). Cần nêu trong phần
      thảo luận; kiểm tra đề cương đã đề cập chưa.
- [x] ~~Chênh ngoài lõi encoder: Mamba không dropout, front-end không CMVN/SpecAugment~~
      — xử lý 2026-10-04 (dropout 0,1 cho Mamba, CMVN toàn cục, SpecAugment).
      Còn nêu khi viết: vị trí dropout trong khối Mamba là lựa chọn của dự án.
- [ ] `best.pt` chọn theo WER val (= `validation` trừ clean-test, D5 — đã
      tách tập từ 2026-10-04); val và clean-test vẫn rải từ cùng video với train
      → lưu ý khi diễn giải.
- [ ] `rtf.py`/`survey.py` chia bucket độ dài theo đề cương (3-30s), không
      khớp dữ liệu thật — RQ2 đã chốt 2026-10-03, việc sửa chuyển lên 🟡 "RQ2".

## ✅ Đã hoàn thành

- [x] **Xác nhận Mamba trên Kaggle T4 (2026-09-24)**, chạy từ local qua Kaggle
      CLI (`scripts/kaggle/verify_mamba.py`): `param_count` in Mamba
      12.292.352 vs Conformer 12.204.288 → **0,72% ĐẠT**. 5 step train batch 16
      (audio 15 s) + eval 16 câu, cả hai encoder chạy qua: Conformer 1,35 s/step,
      VRAM đỉnh 7,76 GiB; Mamba 1,29 s/step, 5,95 GiB. Wheel `causal-conv1d`
      1.5.4 + `mamba-ssm` 2.3.1 (cp312, torch 2.10+cu128) nằm trong output của
      kernel.

- [x] **Push lên GitHub (2026-09-24)**: 6 commit (`f101acb`..`f8d5af4`), gồm
      gỡ 12 file `.docx` khỏi index. File cũ vẫn nằm trong lịch sử git (xem ⚠️).

- [x] ~~Upload tài liệu `.docx` (đề cương, biểu mẫu, notes) lên Google Drive~~
      — **người dùng bỏ qua (2026-09-24)**. Các file chỉ còn trên đĩa local
      (gitignore) + bản cũ trong lịch sử git.

- [x] **Script đo WER clean-test (2026-09-21)**:
      `python -m src.evaluation.eval_clean_test --config configs/model_<x>.yaml`.
      Test thật với 250 câu thật (Conformer, CPU, trọng số ngẫu nhiên: chạy hết
      pipeline, WER vô nghĩa). Còn: chạy với checkpoint thật khi có; Mamba chưa
      test (Kaggle); reference hiện là `pseudo_label` cho cả 250 câu vì
      `corrected_text` chưa hiệu đính (🟢) — script in rõ số câu mỗi loại.

- [x] **`param_count.py` đọc yaml (2026-09-21)**: dùng `build_encoder` của
      `train.py`; test local: Conformer 12.204.288 khớp số đã đo, đổi
      `n_layers` trong yaml thì số đổi theo. Nhánh Mamba chưa chạy (không CUDA).

- [x] **Chuẩn hoá tài liệu gốc (2026-09-19)**: tách `Plan.md` (kế hoạch/trạng
      thái/quyết định/nhật ký), `ARCHITECTURE.md` (sơ đồ, luồng dữ liệu, bản đồ
      file), `docs/CONVENTIONS.md` (quy ước), rút gọn `README.md`, cập nhật
      `CLAUDE.md`. Rà repo public, bổ sung `.gitignore`, gỡ 12 file nhị phân
      khỏi index. *(Đã commit local, chưa push — xem 🟡.)*
- [x] **Training loop thật** (`139f018`): checkpoint/resume tự động
      (`checkpoints/<experiment_name>/{latest,best}.pt`), eval WER định kỳ,
      tensorboard. Test thật với dữ liệu thật trên ConformerEncoder (CPU) — resume
      đúng epoch/global_step. Sửa bug: `lr: 3e-4` bị PyYAML đọc thành string →
      `3.0e-4`. Nhánh Mamba chưa test (xem ⚠️).
- [x] Đề cương chi tiết + tóm lược, đăng ký với GVHD (Tuần 1-2).
- [x] Cấu trúc project + skeleton code toàn bộ pipeline.
- [x] **Rủi ro kỹ thuật lớn nhất**: cài `mamba-ssm` CUDA kernel trên Kaggle T4
      (pin tag v2.3.1), forward pass thật chạy được (Tuần 1).
- [x] Khảo sát thống kê VietSuperSpeech (`src/data/survey.py`), phát hiện sai
      lệch số liệu so với đề cương (Tuần 3).
- [x] Tokenizer BPE tiếng Việt (vocab_size=1000, 60.656 transcript).
- [x] Clean-test manifest 250 câu (seed=42).
- [x] Notebook EDA `notebooks/02_dataset_eda.ipynb` (đã test chạy thật).
- [x] Sửa bug decode audio: cột `audio` là string path, không tự giải mã.
- [x] `src/data/prefetch_audio.py`: tải song song đúng 67.405 file cần dùng,
      resume qua session, lỗi từng file không sập cả batch (test logic xong,
      chưa chạy full — xem ⏸).
- [x] Push code lên GitHub: https://github.com/VuKhah/specialized-thesis
