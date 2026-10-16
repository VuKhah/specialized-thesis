# ĐỀ CƯƠNG KHÓA LUẬN TỐT NGHIỆP (bản điều chỉnh)

> **BẢN NHÁP — chưa duyệt.** Soạn 2026-10-12 để người dùng duyệt, sau đó người dùng tự chuyển sang Word theo
> Mẫu 4. Không thay `docs/de_cuong/*.docx` đã nộp (giữ làm lưu trữ). Dàn mục theo đề cương chi tiết cũ.
> Thông tin cá nhân để trống có chủ đích (repo public) — điền khi chuyển sang Word.
> Chỗ đánh dấu **[CẦN NGƯỜI DÙNG CHỌN]** là quyết định mở, AI chưa chọn. **[chờ GVHD]** là việc xin ý kiến thầy.

- Họ và tên SV: `[...]` — MSSV: `[...]`
- Khoa: Công nghệ Thông tin — Hệ: Đại trà — Chuyên ngành: Trí tuệ Nhân tạo
- Thời gian: Học kỳ 1, năm học 2026-2027
- GV hướng dẫn: `[...]`
- **Tên đề tài (đề nghị đổi):** *Khảo sát Mamba làm encoder cho nhận dạng tiếng nói tiếng Việt hội thoại*
  **[chờ GVHD]** — người dùng đã xin đổi; tên cũ: "Nghiên cứu kiến trúc Mamba và Con-Former Encoder cho Nhận dạng
  tiếng nói trong hội thoại tiếng Việt".

## Nhiệm vụ của khóa luận

1. Tìm hiểu cơ sở lý thuyết: ASR end-to-end với CTC, Conformer, mô hình không gian trạng thái chọn lọc (Mamba/S6)
   và các biến thể hai chiều dùng cho tiếng nói.
2. Huấn luyện từ đầu, trong cùng một pipeline CTC, ba encoder ~12M tham số: **Mamba hai chiều B1** (ExtBiMamba xếp
   chồng), **ConExtBiMamba** (khung Conformer, MHSA thay bằng ExtBiMamba) và **Conformer-12M** làm baseline.
3. Đặt kết quả cạnh nhóm mô hình có pre-train làm tham chiếu: **Parakeet-CTC-0.6B-vi** (fine-tune) và
   **PhoWhisper-small** (zero-shot).
4. Đánh giá theo ba trục: độ chính xác (WER), hiệu quả (tốc độ suy luận, bộ nhớ, chi phí huấn luyện), đặc điểm lỗi;
   nhận xét đánh đổi và khuyến nghị theo kịch bản tài nguyên.
5. Xây dựng demo minh họa (Gradio) và viết khóa luận.

## Phần MỞ ĐẦU

### 1. Tính cấp thiết của đề tài

Encoder ASR phổ biến hiện nay (Transformer, Conformer) dùng self-attention, có chi phí bậc hai theo độ dài chuỗi.
Mamba (Gu & Dao, 2023) là mô hình không gian trạng thái chọn lọc có chi phí tuyến tính, đã được thử cho ASR tiếng
Anh và một số ngôn ngữ khác (arXiv:2405.12609; Miyazaki và cs. 2024; Jiang và cs. 2024). Các khảo sát này cho kết
quả không đồng nhất: ưu thế tốc độ của Mamba chỉ rõ ở audio dài (Jiang và cs. 2024 nêu điểm giao ~60 s), và xếp
chồng Mamba hai chiều thuần có bài cho kết quả kém Conformer (arXiv:2405.12609), có bài cho kết quả tương đương
(Miyazaki và cs. 2024). Trong phạm vi các nguồn đã tra, **chưa tìm thấy** công trình đánh giá Mamba làm encoder ASR
cho tiếng Việt hội thoại (việc tra cứu chưa hệ thống: chưa qua ACL Anthology, Interspeech 2025-26, VLSP). Tiếng
Việt hội thoại (nói nhanh, nuốt âm, nhiễu nền, nhiều phương ngữ) và điều kiện tài nguyên hạn chế (GPU T4) là bối
cảnh thực tế cần một đánh giá có số liệu về đánh đổi độ chính xác – tốc độ – tài nguyên.

### 2. Mục đích của đề tài

Khảo sát Mamba (trọng tâm) làm encoder cho ASR tiếng Việt hội thoại trên VietSuperSpeech: đo độ chính xác, hiệu
quả và đặc điểm lỗi của hai encoder Mamba so với Conformer cùng cỡ huấn luyện từ đầu, đặt cạnh mô hình pre-train
lớn; từ đó nhận xét đánh đổi và đưa khuyến nghị theo kịch bản tài nguyên.

**Mô hình trọng tâm (người dùng chốt 2026-10-12): ConExtBiMamba** (khung Conformer, MHSA → ExtBiMamba, theo
arXiv:2405.12609). Mamba hai chiều xếp chồng thuần (B1) là biến thể đối chứng trong họ Mamba; Conformer-12M là baseline.

**Câu hỏi nghiên cứu**

- **RQ1 — Độ chính xác.** Trong cùng pipeline CTC, cùng ~12M tham số và cùng công thức huấn luyện, hai encoder Mamba
  đạt WER thế nào so với Conformer trên tiếng Việt hội thoại, trên câu từ video đã gặp và video giữ riêng? Kết quả
  nhóm này đứng ở đâu so với mô hình pre-train lớn (chỉ so tham chiếu, khác điều kiện)?
- **RQ2 — Hiệu quả.** Các mô hình khác nhau thế nào về RTF, độ trễ, VRAM khi suy luận, số tham số và GPU-giờ huấn
  luyện; RTF/độ trễ thay đổi thế nào theo độ dài audio (dải độ dài lấy từ audio ghép nhân tạo)?
- **RQ3 — Đặc điểm lỗi.** Các mô hình mắc loại lỗi nào nhiều/ít khác nhau (phân loại lỗi định tính trên clean-test)?

**[CẦN NGƯỜI DÙNG CHỌN] Độ phân giải khung 10 ms so với 40 ms (subsampling 4×)** đặt ở đâu:
- **Phương án A — vào RQ1:** "Thứ hạng WER giữa các encoder có đổi khi độ phân giải khung đổi từ 10 ms sang 40 ms
  không?" (kiểm giả thuyết ConExtBiMamba kém B1 do độ phân giải — `conextbimamba_gap_2026-10-12.md` mục 6-7).
- **Phương án B — vào RQ2:** "Subsampling 4× đổi đánh đổi WER – tốc độ của từng encoder thế nào?" (số đo 08/10:
  subsampling tăng tốc B1 4,0×, Conformer 2,6×/step — `subsample_speed_2026-10-08.md`).
- **Phương án C — RQ riêng (RQ4)**, hoặc nêu ở cả hai, mỗi RQ một khía cạnh.

### 3. Cách tiếp cận và phương pháp nghiên cứu

Nghiên cứu thực nghiệm, so sánh ở **hai mức, không trộn khi kết luận**:

| Mức | Mô hình | Điều kiện | Dùng để |
|---|---|---|---|
| **Nhóm train từ đầu (có kiểm soát)** | Mamba B1 (12.285.696 tham số), ConExtBiMamba (12.241.536), Conformer-12M | Cùng front-end (log-mel 80 / 25 ms / 10 ms + CMVN toàn cục + SpecAugment), tokenizer BPE 1000, CTC head, mã huấn luyện, dữ liệu, công thức; chỉ encoder khác nhau | Kết luận về kiến trúc encoder |
| **Nhóm pre-train (tham chiếu)** | Parakeet-CTC-0.6B-vi (fine-tune), PhoWhisper-small 244M (zero-shot) | Giữ pipeline, front-end, tokenizer, dữ liệu pre-train riêng; chỉ dùng chung chuẩn hóa văn bản trước WER và tập đánh giá | Đặt mốc thực tế, bàn đánh đổi tài nguyên; không kết luận về kiến trúc |

Mamba một chiều không huấn luyện, chỉ trích số liệu của arXiv:2405.12609 (Bảng XII).

**Dữ liệu.** VietSuperSpeech (HF Hub, ghim phiên bản), đo thật 67.405 câu / 245,42 h, câu dài 10-15 s, nhãn do
mô hình Zipformer sinh tự động. Hai bước xử lý (lệch so với đề cương cũ, có lý do ghi trong `dataset_discrepancy.md`):
- **Lọc A1:** bỏ câu nhận dạng ngôn ngữ (Whisper-small) khác tiếng Việt, 149 video phỏng vấn nước ngoài, nhãn có
  < 20% ký tự mang dấu.
- **Giữ riêng theo video:** split `validation` gốc trùng video với `train` (561/562 video), nên giữ riêng ngẫu nhiên
  29 video (~6%, seed 42) khỏi train/val.

| Tập | Câu | Giờ | Video | Vai trò |
|---|---|---|---|---|
| train | 45.442 | 165,18 | 463 | Huấn luyện |
| val | 4.824 | 17,51 | 395 | Chọn checkpoint (video đã gặp) |
| val_unseen | 3.011 | 10,99 | 29 | Đánh giá trên video chưa gặp, nhãn tự động |
| clean-test | 203 | 0,74 | 29 | Đánh giá cuối, nhãn hiệu đính tay, chấm **một lần** |

**Công thức huấn luyện (nhóm train từ đầu).** CTC, không mô hình ngôn ngữ; AdamW, LR 3e-4, warmup 1.000 step,
batch toàn cục 16, DDP 2×T4 + AMP (khối Mamba fp32), speed perturbation 0,9/1,0/1,1, dropout 0,1. Lịch WSD: 30 epoch
LR hằng + **8 epoch giảm LR dạng 1-sqrt**, sau đó **trung bình checkpoint**. Không dùng InterCTC. Các lần chạy:
1. Trung bình 10 checkpoint cuối (epoch 20-29) của lần huấn luyện v1 (LR hằng) cho cả ba mô hình — số tham chiếu.
2. Pha giảm LR 8 epoch rẽ nhánh từ v1 cho B1, ConExtBiMamba, Conformer v1 (không mã hóa vị trí).
3. Conformer có mã hóa vị trí tương đối (cấu hình chuẩn Gulati và cs. 2020) huấn luyện từ đầu cùng công thức; đây
   là baseline Conformer chính, Conformer v1 thành ablation (hiệu số gộp tác dụng PE và sửa padding trước conv).
4. Nhánh **40 ms** (Conv2d subsampling 4×): ba mô hình × 1 seed, từ đầu, cùng công thức.
5. Đo lại RQ2 cho mọi cấu hình.

Bảng chính dùng **1 seed**. Seed thứ hai chỉ thêm khi qua cổng chốt trước: bước 1-5 xong trước ~22/10 **và** còn
≥ 60 GPU-h. Đóng băng thực nghiệm GPU ngày **01/11**.

**Nhóm pre-train.** Parakeet: full fine-tune ~7 GPU-h (≈ chi phí Conformer-12M v1), LR 1e-4 theo config fine-tune
chính thức của NeMo, fp32. PhoWhisper-small: zero-shot.

**Đánh giá.**
- WER giải mã greedy, sau một hàm chuẩn hóa văn bản chung cho mọi mô hình; báo trên val, val_unseen, clean-test.
- Khoảng tin cậy 95% bằng **bootstrap theo khối video** (câu trong cùng video không độc lập); so cặp mô hình bằng
  bootstrap cặp.
- RQ2: RTF và độ trễ batch 1, VRAM đỉnh, số tham số, GPU-giờ huấn luyện; đo mọi mô hình trên cùng một loại máy; dải
  độ dài từ audio ghép các đoạn liên tiếp cùng video (chỉ để đo, không đo WER audio dài).
  **[CẦN NGƯỜI DÙNG CHỌN]** các mức độ dài audio ghép (đang chờ AI đề xuất → người dùng duyệt, `TODO.md` RQ2).
- RQ3: phân loại lỗi định tính trên clean-test.

### 4. Đối tượng nghiên cứu

- Mamba (S6) hai chiều trong vai trò encoder ASR dạng CTC, ở hai cách tổ chức: xếp chồng (B1) và lồng vào khung
  Conformer (ConExtBiMamba); Conformer làm đối chứng.
- Tiếng Việt hội thoại tự nhiên qua VietSuperSpeech.

### 5. Phạm vi nghiên cứu

- ASR end-to-end, giải mã CTC greedy; không decoder attention, không transducer, không rescoring bằng mô hình ngôn
  ngữ.
- Nhóm train từ đầu giới hạn ~12M tham số, hai độ phân giải khung 10 ms và 40 ms.
- Dữ liệu: VietSuperSpeech sau lọc A1 và giữ riêng video (bảng ở mục 3); câu 10-15 s.
- Hạ tầng: Kaggle (2× T4), tối đa 3 tài khoản × 30 GPU-h/tuần (đề cương cũ ghi Colab — lệch có chủ đích).
- Không triển khai lên thiết bị; RTF/độ trễ đo trên GPU/CPU sẵn có.

### 6. Phân tích những công trình có liên quan

- **Mamba** (Gu & Dao, 2023): SSM chọn lọc, quét song song, chi phí tuyến tính theo độ dài.
- **Mamba hai chiều cho ASR:** arXiv:2405.12609 đề xuất ExtBiMamba, ConExtBiMamba hơn Conformer trên
  LibriSpeech-100/960 và vài tập khác, nhưng xếp chồng ExtBiMamba thuần kém xa; Miyazaki và cs. (2024) xếp chồng
  BiMamba thuần (ESPnet, CTC, LibriSpeech-100) đạt 9,1/23,5 so với Conformer 9,8/23,3. Jiang và cs. (2024,
  ConMamba) chỉ thấy lợi thế tốc độ suy luận khi audio > 60 s.
- **Conformer** (Gulati và cs., 2020) với mã hóa vị trí tương đối; Fast Conformer (arXiv:2305.05084): subsampling
  8× không làm tăng WER, encoder nhanh 2,8×.
- **Mô hình pre-train tiếng Việt:** PhoWhisper (arXiv:2406.02555), Parakeet-CTC-0.6B-vi (NVIDIA, ~2.000 h từ 10 bộ).
- **Phương pháp huấn luyện/đánh giá:** SpecAugment (Park và cs., 2019); speed perturbation (Ko và cs., 2015); lịch
  WSD (Hägele và cs., 2024); biến thiên theo seed (Bouthillier và cs., 2021); bootstrap cho WER (Bisani & Ney,
  2004; Liu và cs., 2020).

### 7. Kết quả dự kiến đạt được

- Các checkpoint đã huấn luyện của ba encoder ~12M ở 10 ms (và nhánh 40 ms), mã nguồn pipeline CTC dùng chung.
- Bảng WER kèm khoảng tin cậy bootstrap theo video trên val, val_unseen, clean-test đã hiệu đính; bảng/biểu đồ đánh
  đổi WER – RTF – VRAM – GPU-giờ cho cả hai nhóm.
- Số liệu thực nghiệm về tác động của độ phân giải khung (10 ms / 40 ms) lên từng encoder trên dữ liệu này.
- Phân tích lỗi định tính có ví dụ; tập clean-test 203 câu hiệu đính tay.
- Demo Gradio.
- Nhận xét và khuyến nghị **trong phạm vi đã đo** (một bộ dữ liệu, ~12M tham số, CTC greedy, 1 seed, T4); không
  khái quát ngoài phạm vi này.

**[CẦN NGƯỜI DÙNG CHỌN]** có đưa **kết quả sơ bộ v1** vào đề cương không (WER clean-test trên nhãn giả: B1 17,1%,
ConExtBiMamba 20,3%, Conformer-12M 26,5%, PhoWhisper-small zero-shot 15,5%, Parakeet zero-shot 14,0% —
`conextbimamba_gap_2026-10-12.md` mục 1). Nếu có, phải ghi rõ là nhãn giả, công thức v1 chưa hội tụ.

## Phần NỘI DUNG (ý chính từng chương)

- **Chương 1 — Cơ sở lý thuyết:** ASR, log-mel, CTC; Transformer/Conformer; SSM (S4 → S6/Mamba), Mamba hai chiều;
  subsampling và độ phân giải khung.
- **Chương 2 — Phân tích và thiết kế hệ thống:** khảo sát dữ liệu, lọc A1, giữ riêng video; pipeline CTC dùng chung,
  khớp tham số; thiết kế hai mức so sánh; hạ tầng Kaggle. **[chờ GVHD]** dạng "thiết kế" mong đợi cho đề tài nghiên cứu.
- **Chương 3 — Thực nghiệm và đánh giá:** quá trình huấn luyện (v1 → pha giảm LR, trung bình checkpoint, nhánh
  40 ms), RQ1, RQ2, RQ3; tách bạch nhóm train từ đầu và nhóm pre-train.
- **Chương 4 — Ứng dụng minh họa và tổng kết:** demo Gradio; đối chiếu với công trình liên quan; khuyến nghị theo
  kịch bản tài nguyên.
- **Kết luận:** đóng góp, hạn chế, hướng phát triển (Mamba-3, mô hình ngôn ngữ khi giải mã, nhiều seed, dữ liệu
  khác).

## Hạn chế đã biết

- **Nhãn giả:** nhãn VietSuperSpeech do Zipformer sinh; WER trên val/val_unseen đo so với nhãn giả. Mô hình
  fine-tune trên nhãn này có thể học cả cách viết của Zipformer. Kết luận cuối chỉ dựa trên clean-test đã hiệu đính
  (tiến độ hiệu đính: 11/203 câu ngày 11/10 — đường găng).
- **Clean-test nhỏ:** 203 câu / 0,74 h / 29 video; khoảng tin cậy rộng. Tách theo video không bằng tách theo người
  nói (MC chương trình lớn có thể đã xuất hiện ở video khác).
- **1 seed** cho bảng chính: khoảng tin cậy bootstrap chỉ phản ánh biến thiên do mẫu test, không do huấn luyện.
- **Không dò LR:** LR 3e-4 dùng chung, chọn một lần; có thể có mô hình bị thiệt.
- **Kaggle T4:** không bf16, khối Mamba chạy fp32; số đo tốc độ gắn với phần cứng này; phiên tối đa 9-12 h.
- **Câu ngắn đồng nhất 10-15 s:** dải độ dài cho RQ2 phải lấy từ audio ghép nhân tạo.
- **Nhóm pre-train khác điều kiện:** dữ liệu pre-train, kích thước, subsampling (Parakeet 8×) khác; dữ liệu
  pre-train có thể trùng nguồn với VietSuperSpeech (card không nêu) — chỉ dùng làm tham chiếu.

## KẾ HOẠCH THỰC HIỆN (các tuần còn lại; tuần tính từ thứ Hai)

| Tuần | Ngày | Công việc | Mốc |
|---|---|---|---|
| 8 | 12/10–18/10 | Trung bình checkpoint v1 + eval; khởi chạy pha giảm LR (B1, ConExt, Conformer v1) và Conformer PE tương đối; hoàn thành fine-tune Parakeet; viết đề cương mới; hiệu đính clean-test | Đề cương nháp |
| 9 | 19/10–25/10 | Nhánh 40 ms (3 mô hình); **gặp GVHD lần 2** (đề cương, tên đề tài, kết quả sơ bộ); xét cổng seed 2 | Cổng seed 2 ~22/10 |
| 10 | 26/10–01/11 | Seed 2 (nếu qua cổng); đo RQ2 mọi cấu hình; viết Ch2 | **Đóng băng thực nghiệm GPU 01/11** |
| 11 | 02/11–08/11 | Xong hiệu đính clean-test; chấm clean-test **một lần** mọi mô hình; bootstrap; bảng/biểu đồ | **Đóng băng số liệu 08/11** |
| 12 | 09/11–15/11 | RQ3 phân tích lỗi; viết Ch3 | **Nháp Ch3 15/11** |
| 13 | 16/11–22/11 | Hoàn thiện Ch1, Ch2, Ch3; gặp GVHD | |
| 14 | 23/11–29/11 | Ch4, Mở đầu, Kết luận, TLTK; hoàn thiện mã + demo | **Bản đủ gửi GVHD 27/11** |
| 15 | 30/11–06/12 | Sửa theo góp ý, slide, báo cáo thử | **Nộp** |

Phần bỏ được đầu tiên nếu thiếu thời gian/quota: seed 2, sau đó nhánh 40 ms của ConExtBiMamba/Conformer.
**[CẦN NGƯỜI DÙNG CHỌN]** thứ tự cắt giảm này (AI đề xuất, chưa có quyết định).

## Tài liệu tham khảo (chỉ nguồn đã có trong `docs/notes`)

Đánh dấu **[chưa kiểm]**: notes không ghi đã đọc/đối chiếu bản gốc — phải kiểm trước khi đưa vào Word. Tên bài/tác giả/hội nghị của mọi mục do AI điền theo trí nhớ (notes phần lớn chỉ ghi số arXiv) — đối chiếu lại khi chuyển sang Word.

1. A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023.
2. *Mamba in Speech: Towards an Alternative to Self-Attention.* arXiv:2405.12609, 2024.
   [tên bài + tác giả chưa kiểm — notes chỉ ghi số arXiv]
3. K. Miyazaki và cs. *Exploring the Capability of Mamba in Speech Applications.* Interspeech 2024,
   arXiv:2406.16808. [tên bài chưa kiểm]
4. X. Jiang và cs. *Speech Slytherin: Examining the Performance and Efficiency of Mamba for Speech Separation,
   Recognition, and Synthesis.* arXiv:2407.09732, 2024. [tên bài chưa kiểm]
5. A. Gulati và cs. *Conformer: Convolution-augmented Transformer for Speech Recognition.* Interspeech 2020,
   arXiv:2005.08100.
6. *Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition.*
   arXiv:2305.05084, 2023. [tên bài + tác giả chưa kiểm]
7. D. S. Park và cs. *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Interspeech
   2019, arXiv:1904.08779.
8. T. Ko và cs. *Audio Augmentation for Speech Recognition.* Interspeech 2015.
9. A. Radford và cs. *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). arXiv:2212.04356, 2022.
10. T.-T. Le và cs. *PhoWhisper: Automatic Speech Recognition for Vietnamese.* ICLR 2024 Tiny Paper,
    arXiv:2406.02555. [chưa kiểm]
11. NVIDIA. *Parakeet-CTC-0.6B-vi* — model card Hugging Face (đọc 2026-10-11), NVIDIA Open Model License.
12. thanhnew2001. *VietSuperSpeech* — dataset Hugging Face (đã kiểm lịch sử commit 2026-09-14).
13. A. Hägele và cs. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* arXiv:2405.18392,
    2024. [chưa kiểm]
14. X. Bouthillier và cs. *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021, arXiv:2103.03098.
15. M. Bisani, H. Ney. *Bootstrap Estimates for Confidence Intervals in ASR Performance Evaluation.* ICASSP 2004.
    [chưa kiểm]
16. Liu và cs. arXiv:1912.09508 (bootstrap theo khối cho WER). [chưa kiểm — notes không ghi tên bài]
17. Zevallos và cs. Interspeech 2025 (ConMamba so Conformer, 9 ngôn ngữ ít tài nguyên). [chưa kiểm]

Thiếu trong notes, cần bổ sung khi viết: bài gốc CTC (Graves và cs., 2006); S4. Ba tài liệu của đề cương cũ
(Samba-ASR arXiv:2501.02832; arXiv:2506.16969; arXiv:2410.00070) chưa có ghi chú đối chiếu — **[chưa kiểm]**, chỉ giữ
nếu thực sự trích dẫn.

*Ngày ... tháng ... năm 2026 — Người viết đề cương / Ý kiến của giáo viên hướng dẫn (ký, ghi rõ họ tên).*
