# Ứng viên mô hình cho khảo sát + nhãn tiếng Anh (A1)

Ghi 2026-10-01, sau buổi GVHD lần 1 (`gvhd_buoi_1.md`). **Trạng thái: AI tra
cứu và đề xuất — chưa chốt.** Danh sách mô hình và hướng xử lý A1 do **người
dùng** quyết (thầy giao "tự đọc rồi làm", dặn "đừng ôm đồm").

## 1. Ứng viên đã tra (model card Hugging Face, đọc 2026-10-01)

| Mô hình | Kiến trúc | Tham số | Pre-train / huấn luyện trên | Có tiếng Anh? | License | Ghi chú |
|---|---|---|---|---|---|---|
| `nvidia/parakeet-ctc-0.6b-Vietnamese` | FastConformer-CTC (Conformer cải tiến, subsampling 8×) | ~600M | ~2.000 h, 10 bộ tiếng Việt (+ Common Voice 20.0, MS-SNSD) | **Có** — tên card: "Parakeet-CTC-0.6B **Unified Vietnamese–English CS**" (code-switching); đính chính 2026-10-02 | NVIDIA Open Model License (cho phép thương mại & phi thương mại) | **Đúng ô "Conformer có pre-train"**. Avg WER 9,30% blind test. Không nêu VietSuperSpeech. Cần NeMo; card không có link `.nemo` trực tiếp. *Mâu thuẫn:* trang build.nvidia.com mô tả ">4.000 h", "Vietnamese-English" — chưa đối chiếu được |
| `vinai/PhoWhisper-{tiny,base,small,medium,large}` | Whisper (encoder-decoder Transformer) | 39M / 74M / 244M / 769M / 1,55B | Whisper đa ngôn ngữ, fine-tune 844 h tiếng Việt (CMV-Vi, VIVOS, VLSP 2020, dữ liệu riêng) | **Có** — kế thừa Whisper gốc (phần lớn dữ liệu pre-train là tiếng Anh; *tỉ lệ cụ thể cần lấy từ bài Whisper*) | xem card | Dùng `transformers`; LoRA vừa T4. Ghi chú cũ: `de-tai-phowhisper-vietsuperspeech.md` |
| `nguyenvulebinh/wav2vec2-base-vietnamese-250h` | wav2vec 2.0 (Transformer encoder, SSL) + CTC | 95M (large: 317M) | SSL 13.000 h YouTube tiếng Việt không nhãn; fine-tune 250 h VLSP | Không nêu | **CC BY-NC 4.0** (phi thương mại — đủ cho khóa luận) | Gần pipeline CTC hiện tại nhất. VIVOS 10,77% WER không LM |
| `kyle/vi-asr-fastconformer-114m` | FastConformer-**Transducer** | 114M | từ `stt_en_fastconformer_transducer_large` (tiếng Anh) → ~14 bộ tiếng Việt | Pre-train gốc tiếng Anh | CC-BY-4.0 | ⚠️ **Đã train trên VietSuperSpeech** (báo WER 21,81% trên bộ này) → rò rỉ dữ liệu, **không dùng làm đối chứng** |
| `hynt/Zipformer-30M-RNNT-6000h` | Zipformer (Conformer cải tiến) RNNT, k2 | ~30M | 6.000 h tiếng Việt | Không | **CC-BY-NC-ND-4.0** (cấm tác phẩm phái sinh) | ⚠️ **Chính là mô hình sinh nhãn VietSuperSpeech** → đo WER trên nhãn của nó là vòng; ND cấm fine-tune phân phối. **Loại** (chỉ nhắc ở phần dữ liệu) |
| Mamba ASR pre-train | — | — | Chưa tìm thấy checkpoint Mamba ASR tiếng Việt. MLMA (arXiv 2510.18684, ConMamba-CTC 32-42M) chỉ 6 tiếng châu Âu; Samba-ASR chỉ tiếng Anh | — | — | → **Mamba train từ đầu** là hướng khả thi duy nhất (code đã có) |

Nguồn: các trang `huggingface.co/<tên mô hình>` ở trên; MLMA
`arxiv.org/html/2510.18684v1`; Samba-ASR `arxiv.org/abs/2501.02832`; ConMamba
(Speech Slytherin) `arxiv.org/abs/2407.09732`.

## 2. A1 — nhãn tiếng Anh phiên sai, xét theo ngôn ngữ pre-train

~~Nhãn do Zipformer chỉ-tiếng-Việt sinh ra → đoạn tiếng Anh bị phiên thành âm
tiết tiếng Việt vô nghĩa.~~ **Đính chính 2026-10-02 (xem mục 4):** đọc nhãn thật
thì các câu bị đánh dấu là **chuỗi từ giống tiếng Anh nhưng sai** (vd. "AN MINI
BISUMBASIO VIENTAL A WAX BIN MOON OF THE THOP INVESTMENT…"), không phải âm tiết
Việt. Kết luận bên dưới không đổi: nhãn vẫn sai, mô hình nghe đúng tiếng Anh vẫn
bị chấm sai. Bảng dưới đọc "âm tiết Việt vô nghĩa" là "chuỗi giả tiếng Anh".
(heuristic: ~19,4% câu train, 23,2% clean-test; chưa nghe kiểm).

| Nhóm mô hình | Khi **train/fine-tune** trên nhãn này | Khi **đánh giá** trên nhãn này |
|---|---|---|
| Pre-train có tiếng Anh (PhoWhisper/Whisper, Parakeet Việt–Anh CS) | Bị dạy "quên" tiếng Anh: học phiên tiếng Anh thành âm tiết Việt vô nghĩa → hại đúng thế mạnh của mô hình | Mô hình ra từ tiếng Anh **đúng** nhưng bị chấm **sai** → WER bị thổi phồng, thiệt cho nhóm này |
| Pre-train chỉ tiếng Việt (wav2vec2-vi; ~~Parakeet-vi~~ — Parakeet thuộc nhóm có tiếng Anh, xem bảng mục 1) | Nhiễu nhãn thông thường | Ít lệch hơn — tự nhiên cũng ra âm tiết Việt |
| Train từ đầu (Mamba, Conformer nhỏ) | Nhiễu nhãn; học bắt chước luôn lỗi của Zipformer | Ít lệch, nhưng WER thấp ở đoạn này **không** phản ánh nhận dạng đúng |

→ Với khảo sát nhiều mô hình, nhãn tiếng Anh sai **không còn trung tính**: nó
thiên vị chống lại mô hình biết tiếng Anh. Phương án (từ `TODO.md`):
(a) giữ, nêu hạn chế · (b) giữ train, báo WER hai mức (toàn bộ / chỉ câu tiếng
Việt) · (c) lọc khỏi train + clean-test.
**AI nghiêng về (c) cho train + (b) cho báo cáo** (lọc khỏi train/fine-tune;
clean-test báo cả hai mức để vẫn bàn được hiện tượng code-switching). Phải kiểm
heuristic bằng cách nghe một mẫu trước khi lọc. **Người dùng quyết; nếu lọc thì
phải chốt trước khi tạo dataset/train.**

Hệ quả phụ: các mô hình ra chữ hoa/dấu câu (Parakeet, PhoWhisper) → cần **một
bước chuẩn hóa văn bản chung** (lowercase, bỏ dấu câu, chuẩn số) trước khi tính
WER cho mọi mô hình.

## 3. Đề xuất đội hình "không ôm đồm" — **người dùng chốt 2026-10-02: đủ 5 mô hình (gồm #5 wav2vec2-base-vi)**

| # | Vai trò | Mô hình | Cách dùng | Chi phí ước (chưa đo) |
|---|---|---|---|---|
| 1 | **Trọng tâm** | Mamba-CTC ~12M | Train từ đầu (code có sẵn) | ~16 h Kaggle (benchmark D8) |
| 2 | **Baseline** | Conformer-CTC ~12M | Train từ đầu, cùng pipeline (code có sẵn) | ~23 h |
| 3 | Conformer pre-train | Parakeet-CTC-0.6B-vi | Zero-shot; fine-tune ngắn nếu còn quota | Zero-shot: vài giờ inference. Fine-tune 600M trên T4: nặng, cần thử |
| 4 | Pre-train đa ngôn ngữ | PhoWhisper-small (244M) | Zero-shot + fine-tune LoRA | Cần ước |
| (5) | Tùy chọn | wav2vec2-base-vi (95M) | Fine-tune CTC | Cần ước |

Lý do: #1-#2 đã gần xong code, giữ được một cặp so sánh có kiểm soát (cùng cỡ,
cùng dữ liệu) làm "lõi" — trả lời "Mamba vs Conformer khi không có pre-train".
#3-#4 trả lời ý thầy "Conformer phải pre-train mới tốt" và cho trục đánh đổi
(độ chính xác vs tham số/tốc độ/VRAM/chi phí). Zero-shot trước vì rẻ (chỉ
inference) và tự nó đã là một kết quả.

Rủi ro: cài NeMo trên Kaggle (#3) chưa thử; quota 2 tài khoản × 30 GPU-h/tuần;
mô hình #3/#4 có subsampling nên RTF không so "cùng điều kiện" với #1/#2 — đúng
tinh thần "không cần giống nhau", nhưng phải nói rõ khi bàn RQ2.

## 4. Ngưỡng "< 20% từ có dấu" dựa vào đâu (người dùng hỏi 2026-10-02)

**Không có căn cứ từ tài liệu.** Đây là heuristic AI đặt ra ở phiên 2026-09-30,
chưa kiểm chứng. Lập luận gốc chỉ là chính tả: từ tiếng Anh **không bao giờ** có
dấu thanh/dấu phụ (ă â ê ô ơ ư đ), còn câu tiếng Việt thì phần lớn âm tiết có dấu
(chỉ âm tiết thanh ngang không chữ phụ như "em", "anh", "con", "ra" là không
dấu). Tỉ lệ từ có dấu do đó là dấu hiệu về **ngôn ngữ của nhãn**, không phải của
audio.

Phân bố trên toàn bộ 60.656 nhãn train (đo 2026-10-02, hàm: NFD rồi đếm ký tự
dấu kết hợp, `đ` tính là có dấu):

| Tỉ lệ từ có dấu | 0-10% | 10-20% | 20-30% | 30-40% | 40-50% | 50-60% | 60-70% | 70-80% | 80-90% | 90-100% |
|---|---|---|---|---|---|---|---|---|---|---|
| Số câu | 10.992 | 792 | 193 | 100 | 46 | 129 | 686 | 5.948 | 25.579 | 16.191 |

Điều số liệu **chứng minh được:**
- Phân bố **hai đỉnh** rõ rệt, đáy ở 40-50% (46 câu): có hai nhóm nhãn tách
  biệt, không phải một dải liên tục.
- **Ngưỡng ít quan trọng:** đặt ngưỡng ở 10% → 10.992 câu; 20% → 11.784; 50% →
  12.123. Dời ngưỡng từ 10% lên 50% chỉ đổi ~1.100 câu (~1,9% dữ liệu).
- Đọc mẫu ngẫu nhiên: nhóm 0-20% là chuỗi giả tiếng Anh vô nghĩa; nhóm 60-70%
  là câu tiếng Việt chen từ tiếng Anh hợp lệ (LIVESTREAM, YOUTUBE, BRAND) — đúng
  là **không** nên lọc nhóm này.

Điều số liệu **không chứng minh được:**
- Rằng **audio** là tiếng Anh. Có thể là tiếng Anh thật, nhưng cũng có thể là
  nhạc, nhiều người nói chồng, tiếng ồn mà Zipformer "bịa" ra chuỗi Latin.
  Với mục đích lọc thì cả hai trường hợp đều là nhãn hỏng, nhưng **khi viết
  khóa luận không được gọi là "câu tiếng Anh"** nếu chưa nghe kiểm — gọi là
  "nhãn không phải tiếng Việt / nhãn hỏng".
- Câu rất ngắn cho tỉ lệ dao động mạnh (vd. "OK NHÁ" = 50%).
- Không có độ chính xác/độ phủ (precision/recall) nào cho đến khi nghe kiểm.

**Cách kiểm chứng rẻ nhất:** việc 🟢 hiệu đính 250 câu clean-test đằng nào cũng
phải nghe → khi nghe, ghi thêm một cột nhãn ngôn ngữ audio (Việt / Anh / chen /
nhạc-ồn). 58 câu bị đánh dấu + 192 câu không bị đánh dấu → có ngay
precision/recall của ngưỡng trên mẫu ngẫu nhiên có seed. Nếu cần chắc hơn, nghe
thêm ~50 câu train ở vùng 10-50% (vùng biên). Đây là căn cứ sẽ trích trong
Chương 3.

## 5. Ghi chú từ lần chạy thử zero-shot local (2026-10-02, CPU, 4 câu — quá ít để kết luận WER)

- Cả 3 mô hình tải được ẩn danh; license: Parakeet NVIDIA Open Model License,
  PhoWhisper-small BSD-3-Clause, wav2vec2 card ghi CC BY-NC 4.0 nhưng repo kèm
  file CC-BY-NC-SA-4.0 (mâu thuẫn, cả hai đều phi thương mại — đủ cho khóa luận).
- Card Parakeet liệt kê phần cứng Ampere/Blackwell/Hopper/Volta, **không có
  Turing (T4)** — cần xác nhận chạy được trên Kaggle T4.
- Câu clean-test index 26 (nhãn giả tiếng Anh): cả 3 mô hình đều ra âm tiết
  Việt — một mẫu, chưa đủ nói gì về A1.
- Thiết lập đo đã chọn: greedy, **không LM 4-gram** (cùng điều kiện Mamba/
  Conformer); trọng số fp32 + autocast fp16, front-end fp32; batch 1, 3 câu
  warm-up; RTF không tính đọc đĩa; VRAM = `max_memory_allocated`. Chữ số giữ
  nguyên (nhãn không có chữ số) — kernel đếm số hyp có chữ số để lượng hóa lỗi
  định dạng số.
