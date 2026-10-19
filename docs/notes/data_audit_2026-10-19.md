# Rà soát xử lý dữ liệu — 2026-10-19

Rà lại code `src/data/`, `src/features/`, `src/tokenizer/`, phần dữ liệu của `src/training/train.py`, và **chạy kiểm
trên dữ liệu thật** (manifest + metadata HF ở `HF_REVISION`, cache local). Script kiểm ở scratchpad phiên (ngoài repo);
mô tả đủ để dựng lại ở dưới.

## 1. Đúng — đã kiểm bằng dữ liệu

| Kiểm | Kết quả |
|---|---|
| Giao nhau theo (split, index): train / val / val_unseen / clean-test | 0 ở mọi cặp |
| Video giữ riêng ∩ video train, val | 0; clean-test và val_unseen ⊂ 29 video giữ riêng; 7 câu/video × 29 = 203 |
| Câu `excluded.tsv` (A1) lọt vào val_unseen / clean-test | 0 |
| Manifest ↔ HF (đường dẫn audio, duration, nhãn giả clean-test) | 0 lệch |
| `source` (HF) ↔ `video_of(audio)` | ánh xạ 1–1 (chỉ khác đuôi `.mp4`) → chia shard theo `source` và giữ riêng theo tên file nhất quán |
| Corpus tokenizer `train_transcripts_filtered.txt` | đúng multiset nhãn của 45.442 câu train sau lọc (không chứa video giữ riêng) |
| Tokenizer BPE1000 | 0 `<unk>`, encode→decode khứ hồi đúng 100% ở cả 4 shard, val, val_unseen; blank = 0, vocab 1001 |
| Khả thi CTC (token + số cặp lặp ≤ số khung), xét cả speed 0,9/1,0/1,1 | 0 câu bất khả ở 10 ms lẫn 40 ms; tỉ lệ nhãn/khung tối đa 0,36 ở 40 ms × 1,1 → `zero_infinity` không âm thầm bỏ câu nào |
| Duration metadata ↔ số mẫu file thật (3.000 file ngẫu nhiên) | lệch 0,000 s; 100% 16 kHz mono |

Đọc code, không thấy lỗi: `feat_lengths = ⌊N/160⌋ + 1` đúng với `center=True`; công thức độ dài `Conv2dSubsampling4`
đúng và khung ra cuối không chạm padding; padding được xoá trước depthwise conv (ConExt, relPE), Mamba đảo theo độ dài
thật; SpecAugment che thời gian chỉ trong khung thật; hệ số speed perturbation là hàm thuần (seed, epoch, idx) và
sampler dùng cùng chỉ số với `__getitem__`; CMVN tính trên train đã lọc.

Lệch nhỏ đã biết, không phải lỗi: Conformer không PE (torchaudio) không xoá padding trước conv (đã ghi ở
`conextbimamba_encoder.py`); front-end tính trên waveform đã đệm 0 nên ~1 khung cuối của câu ngắn trong batch khác đôi
chút so với chạy riêng (reflect-pad của STFT) — ảnh hưởng ≤ 1 khung, cả ba encoder như nhau.

## 2. Phát hiện mới — trùng nội dung giữa video giữ riêng và train

Tách theo **tên video** không tách được **nội dung**: vài video giữ riêng là tập tổng hợp / tập đặc biệt / bản podcast
dùng lại audio của tập thường nằm trong train, cộng phần giới thiệu chương trình lặp lại mỗi tập.

Cách đo: một câu bị đánh dấu nếu ≥ 50% 8-gram từ của nhãn giả xuất hiện trong nhãn train + val (sau lọc).

- **clean-test: 11/203 câu** (6 câu trùng nguyên văn một nhãn train). val_unseen + clean-test: 61/3.214.
- Tập trung ở: `Truong_thanh_la_vuot_qua_nhung_thu_thach_tu_ben_trong_EduStation_-_Tap_dac_biet` (31/65 câu — tập ghép
  từ các tập EduStation khác), `Khui_het_12_tap_trong_12_phut_Have_A_Sip_-_After_Hours_dac_biet` (6/36),
  `Tren_tinh_ban_duoi_tinh_yeu..._Yeu_Lanh_EP5` (9/165, trùng bản Lofi Podcast), `Hay_yeu_nghe..._Tram_Nam_San_Khau_Ep7`
  (6/154), intro "KHÁNG THƯƠNG LÀ MỘT CHƯƠNG TRÌNH…" (Khang_Thuong).
- LM chỉ loại câu trùng **nguyên văn** (`lm_decode.build_lm_text`), nên các câu trùng một phần vẫn ở trong văn bản LM.

**Độ nhạy** (chỉ để đánh giá mức ảnh hưởng — số chính thức giữ nguyên; tính từ
`reports/results/clean_test_scored_2026-10-16_chung/per_sentence.tsv`, WER 203 câu khớp bảng chính):

| Hệ thống | WER 203 | WER 11 câu trùng | WER 192 câu | Δ |
|---|---|---|---|---|
| relPE 40 ms s42 greedy | 0,1211 | 0,0897 | 0,1227 | +0,16 điểm |
| B1 40 ms s42 greedy | 0,1264 | 0,0661 | 0,1296 | +0,32 |
| ConExt 40 ms s42 greedy | 0,1309 | 0,0739 | 0,1339 | +0,30 |
| B1 10 ms | 0,1432 | 0,0897 | 0,1460 | +0,28 |
| relPE 10 ms | 0,1694 | 0,1170 | 0,1721 | +0,27 |
| relPE / B1 / ConExt 40 ms + LM | 0,1024 / 0,1115 / 0,1081 | 0,049 / 0,029 / 0,051 | 0,1052 / 0,1158 / 0,1111 | +0,28 / +0,43 / +0,30 |
| Parakeet zero-shot | 0,1378 | 0,1012 | 0,1398 | +0,19 |
| PhoWhisper zero-shot | 0,1486 | 0,1012 | 0,1510 | +0,25 |
| Parakeet FT | 0,0943 | 0,0682 | 0,0956 | +0,14 |

Nhận xét:
- Mức thay đổi 0,1–0,4 điểm WER; thứ hạng trong từng bảng không đổi. 11 câu này vốn dễ hơn (mô hình pre-train không
  thấy chúng vẫn đạt ~10%), nên chỉ phần chênh thêm của nhóm train từ đầu (~3–7% so với ~10%) mới là dấu hiệu đã gặp nội
  dung.
- Nhóm train từ đầu được lợi nhiều hơn nhóm pre-train zero-shot (Mamba nhiều hơn relPE một chút: +0,32 so với +0,16 ở
  seed 42). Đây là **suy luận** từ 11 câu, chưa có kiểm định; chưa chạy lại bootstrap/kiểm định dấu trên 192 câu.
- Không ảnh hưởng tới RQ2 (thời gian/VRAM) và kết luận "không phân biệt được" ở 40 ms gần như chắc chắn giữ — nhưng cần
  bootstrap lại để nói điều đó có số.

## 3. Quyết định

**Chốt 19/10 (người dùng):** bỏ qua 11 câu — giữ nguyên 203 câu, không sửa tập, không khai báo trong khoá luận (ảnh hưởng 0,1–0,4 điểm, thứ hạng không đổi). Các phương án dưới đây giữ để tham khảo, không làm.

1. Báo cáo thế nào: (a) giữ 203 câu, khai hạn chế + bảng độ nhạy ở phụ lục; (b) thêm cột/phân tích 192 câu làm kiểm
   tra độ vững (cùng bootstrap cặp theo video); (c) khác. Không đổi tập chính thức khi chưa chốt.
2. Có đưa script kiểm (rò rỉ split, 8-gram, tokenizer/CTC, duration) vào repo (`src/data/audit.py`) để tái lập không.
3. Câu chữ Ch3 về "độc lập theo video": nên viết "độc lập theo video, không loại được tập tổng hợp dùng lại nội dung".

## 4. Tiếng Anh chêm trong nhãn giả ("dịch mù") — đối chiếu `Argon_feedback.md`

`Argon_feedback.md` là lời khuyên chung của một chatbot, viết khi chưa biết thiết kế đề tài; đo lại trên dữ liệu train
(45.442 câu sau lọc, nhãn giả ở `HF_REVISION`):

| Đo | Kết quả |
|---|---|
| Bộ lọc A1 loại gì | 50,3 h / 245,4 h: **48,2 h là cả video phỏng vấn khách nước ngoài** (luật `video`), 2,1 h theo LID, chỉ **3 câu** loại riêng vì nhãn |
| Câu có chêm tiếng Anh còn trong train | 18.607 câu (40,9%) có ≥ 1 từ không phải âm tiết Việt; 40.828 từ (1,73% số từ) |
| Từ tiếng Anh phổ biến | Zipformer viết **nguyên chữ**: OK 1.954, MARKETING 377 (0 lần dạng "ma kết tinh"), PODCAST 363, STARTUP 150, CEO 87, KPI 25 |
| Từ tiếng Anh hiếm | 8.223 loại xuất hiện ≤ 2 lần (9.504 lần), nhiều chuỗi méo: *SOFC, ITACH, OVERTINGKING, INDIATION*; phiên âm sai lặp lại: *CON SUMER* ×9, *IN DIPENCE* ×3 |
| Viết tắt | không nhất quán: dạng liền (*MC, MV, TV*) lẫn chữ cái rời (*B* 215, *C* 131, *Z* 111, *X* 80, *D* 79) |

Kết luận:
- "Dịch mù" là **lỗi nhãn giả ở tiếng Anh hiếm/tên riêng**, không phải do pipeline xoá câu chêm hay Việt hoá từ phổ biến.
  Khớp số đã có: nhãn giả sai 37% từ ngoại trên clean-test, từ ngoại 3,1% từ ref nhưng ~21% lỗi của nhóm 40 ms, mô hình
  học nhãn giả nhại lỗi 15–24% (note rq3 mục 4, 6).
- Các khuyến nghị của Argon mà pipeline **đã làm**: giữ câu chêm, nhãn không có chữ số, NFC, BPE1000 với
  `character_coverage=1.0` (0 `<unk>`), SpecAugment + speed perturbation, LM ngoài (beam + từ điển + LM: −13–15 điểm WER
  trên từ ngoại).
- Khuyến nghị **không áp dụng** (luật cứng CLAUDE.md, review 15/10 mục 3): sửa nhãn toàn bộ tập train, đổi vocab, CTC/
  attention lai — đều đổi công thức chung, cần train lại. Các con số "cần 1.000–10.000 h để train từ đầu", "speed
  perturbation biến 170 h thành 510 h", "độ chính xác 92–95% sau 10–20 epoch" không có nguồn và không khớp dữ liệu repo
  (165 h, train từ đầu đạt WER clean-test ~12%; speed perturbation chỉ tạo biến thể của cùng nội dung).
- Việc còn lại là **viết**: khai hạn chế "nhãn giả sai tiếng Anh hiếm và viết tắt không nhất quán" bằng các số trên ở Ch3
  (RQ3) và phần hạn chế — người dùng quyết có đưa bảng này vào không.
