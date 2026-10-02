# Buổi báo cáo GVHD lần 1 — ý kiến và hướng triển khai

Ghi 2026-10-01. **Gặp ở Tuần 6/15** (người dùng đính chính: đang ở Tuần 6, không phải 4-5). **Buổi sau: dự kiến Tuần 9.** Trạng thái: **đã ghi
ý kiến, chưa chốt thiết kế mới, chưa đổi code.** Đây là thay đổi định hướng
lớn nhất từ đầu dự án — mọi quyết định cụ thể ở mục 4 thuộc người dùng/GVHD.

## 1. Ý kiến của thầy (người dùng thuật lại, giữ nguyên ý)

1. **Lệch hướng ngay từ đầu.** Đề tài là *nghiên cứu dạng so sánh các kiến
   trúc cho dữ liệu tiếng Việt*; bản chất là bài toán ASR **và fine-tune**.
   Cần chọn **một mô hình chính (Mamba)** rồi so sánh nó với các kiến trúc
   khác.
2. **So sánh không cần giống nhau toàn bộ.** Mamba bản chất là dạng tuyến
   tính; Conformer là mô hình nhiều tham số, **phải pre-train** (học thật
   nhiều) mới cho kết quả tốt.
3. **Viết nhận xét dựa trên kết quả.** Ví dụ: Mamba kém chính xác hơn 10%
   nhưng nhanh gấp 2 → "chấp nhận đánh đổi độ chính xác lấy tốc độ"; hoặc "tài
   nguyên hạn chế thì dùng mô hình A, tài nguyên dư dả thì dùng B".
4. **Nhiều phương án so sánh hơn**: (1) Mamba, (2) Conformer có pre-train,
   (3)-(4) một vài mô hình khác, (5) một baseline. Khóa luận là **khảo sát**
   nên chỉ cần như vậy.

## 2. AI diễn giải — cần người dùng xác nhận là hiểu đúng ý thầy

- Đề tài chuyển từ **thí nghiệm có kiểm soát** (matched-parameter, chỉ hoán đổi
  encoder, cả hai train từ đầu) sang **khảo sát so sánh nhiều mô hình**, Mamba
  là trọng tâm. Ràng buộc "chỉ encoder khác nhau" trong `CLAUDE.md`/
  `ARCHITECTURE.md` **không còn là điều kiện sống còn** — nhưng chưa bỏ cho tới
  khi chốt.
- Các mô hình đối chứng được phép khác cấu hình, khác số tham số, **dùng
  checkpoint pre-train rồi fine-tune** trên VietSuperSpeech.
- Kết quả cần nhiều trục để bàn đánh đổi: WER (độ chính xác), RTF/độ trễ (tốc
  độ), VRAM, số tham số, chi phí train/fine-tune (GPU-giờ). Chương 3-4 phải có
  phần **khuyến nghị theo kịch bản tài nguyên**.

## 3. Phần việc đã làm — còn dùng được không

| Đã có | Hướng mới |
|---|---|
| Dữ liệu: khảo sát, 4 shard + val, kernel tạo dataset tar, clean-test 250 câu | **Dùng lại hết** — mọi mô hình đều fine-tune/đánh giá trên cùng dữ liệu, cùng clean-test |
| Pipeline CTC dùng chung + `MambaEncoder` + tokenizer BPE + train loop, DDP/AMP benchmark | Dùng cho **Mamba-CTC train từ đầu** (nếu Mamba giữ train từ đầu — câu Q-c) |
| `ConformerEncoder` 12,2M train từ đầu | Có thể thành **baseline** hoặc bỏ — câu Q-d |
| Đo RTF, WER, phân loại lỗi (`rtf.py`, `eval_clean_test.py`) | Mở rộng cho mô hình ngoài (HF/NeMo) — cần lớp bọc chung cho inference |
| Khớp tham số, Mamba hai chiều (A4), nghi vấn biến gây nhiễu | Bớt quan trọng vì không còn yêu cầu "giống hệt"; vẫn là lựa chọn thiết kế cho Mamba |

## 4. Câu hỏi cần làm rõ (chưa trả lời — không tự chọn)

- **Q-a. Thầy có trả lời các câu đã chuẩn bị không** (A1 nhãn tiếng Anh, A2
  RQ2, A3 val/test không độc lập, A4 một/hai chiều, B1, C1-C4)? Nếu không, vẫn
  giữ 🔴.
- **Q-b. Danh sách mô hình cụ thể** (thầy chỉ nêu khung "Mamba · Conformer
  pre-train · 2-3 mô hình khác · baseline"). Thầy có gợi ý tên nào không?
- **Q-c. Mamba train từ đầu hay fine-tune?** Chưa thấy checkpoint Mamba ASR
  pre-train tiếng Việt (AI chưa tra kỹ). Ý 2 của thầy gợi ý Mamba có thể train
  từ đầu, còn Conformer dùng pre-train — cần xác nhận.
- **Q-d. Baseline là gì?** Ví dụ: Conformer nhỏ train từ đầu (đã có code),
  mô hình pre-train dùng nguyên không fine-tune (zero-shot), hoặc mô hình cổ
  điển (BiLSTM-CTC).
- **Q-e. Fine-tune hay chỉ đánh giá zero-shot** các mô hình pre-train lớn?
  Fine-tune mô hình hàng trăm triệu tham số trên 245 h bằng T4 rất tốn quota.
- **Q-f. Đề cương & tên đề tài**: không sửa `.docx` đã nộp (luật repo). Có cần
  nộp đề cương điều chỉnh / đổi tên đề tài theo hướng "khảo sát" không? Hỏi
  GVHD.
- **Q-g. RQ mới**: RQ1-RQ3 cũ (WER · RTF theo độ dài · phân loại lỗi) có giữ
  làm trục so sánh cho tất cả mô hình không?

## 5. Ứng viên mô hình — AI liệt kê theo trí nhớ, **chưa kiểm chứng**

> **Đã tra lại 2026-10-01 → `survey_model_candidates.md`** (bảng dưới là bản trí nhớ ban đầu, giữ để đối chiếu; bản mới có 2 phát hiện: `kyle/vi-asr-fastconformer-114m` đã train trên VietSuperSpeech, Zipformer là mô hình sinh nhãn + license ND).

Phải tra Hugging Face/bài gốc (tên, license, số tham số, có hỗ trợ tiếng Việt,
chạy được trên T4) trước khi đưa vào đề xuất.

| Nhóm | Ứng viên | Ghi chú |
|---|---|---|
| Conformer pre-train | Họ Conformer/FastConformer của NVIDIA NeMo (cần tìm checkpoint có tiếng Việt) | Đúng "Conformer có pre-train" thầy nêu; NeMo trên Kaggle cần thử cài |
| Encoder-decoder pre-train | PhoWhisper (VinAI, tiny→large, fine-tune Whisper cho tiếng Việt) / Whisper gốc | Đã có ghi chú cũ: `de-tai-phowhisper-vietsuperspeech.md`; fine-tune LoRA vừa T4 |
| Tự giám sát + CTC | wav2vec2 tiếng Việt (vd. bản fine-tune 250 h của cộng đồng) | Transformer encoder + CTC, gần pipeline hiện tại nhất |
| Baseline | Conformer nhỏ / BiLSTM-CTC train từ đầu, hoặc pre-train zero-shot | Q-d |

**Cạm bẫy:** nhãn VietSuperSpeech là pseudo-label do
**Zipformer-30M-RNNT-6000h** sinh (`de-tai-phowhisper-vietsuperspeech.md`).
Đưa chính mô hình này vào so sánh thì WER trên `validation`/clean-test chưa
hiệu đính sẽ **lạc quan một cách vòng vo** (tự so với nhãn của mình). Nếu dùng,
chỉ báo WER trên phần clean-test đã hiệu đính tay — càng làm việc 🟢 hiệu đính
250 câu trở nên quan trọng.

## 6. Tác động tới việc đang làm

- **Không bị ảnh hưởng, làm tiếp được:** kernel tạo 5 dataset (bước 5 `TODO.md`),
  hiệu đính clean-test, chương 1 phần cơ sở lý thuyết ASR/CTC/Mamba.
- **Nên chờ chốt mục 4:** train full (bước 6), sửa code dùng chung cho
  Conformer (bước 3 phần riêng Conformer), Mamba hai chiều (A4), viết lại RQ ở
  Chương 1.
- **Quota Kaggle** (2 tài khoản × 30 GPU-giờ/tuần) giờ phải chia cho nhiều mô
  hình → cần ước lượng chi phí từng mô hình trước khi chốt danh sách.

## 7. Trả lời bổ sung (người dùng thuật lại, 2026-10-01)

- **Q-a / A1 (nhãn tiếng Anh):** thầy hỏi ngược: *model pre-train được
  pre-train trên ngôn ngữ nào? Nếu có tiếng Anh thì nhãn tiếng Anh (phiên sai)
  ảnh hưởng gì?* — **"về tự đọc rồi làm"**: thầy giao người dùng tự tìm hiểu và
  quyết. Quyết định thuộc **người dùng** (không còn chờ GVHD); AI chỉ chuẩn bị
  phân tích theo từng mô hình. A2, A3, A4, B1, C1-C4: không thấy nhắc → coi như
  chưa trả lời.
- **Q-b (danh sách mô hình):** thầy **không gợi ý** mô hình. Thầy dặn:
  **"Đừng ôm đồm hết vào người"** → danh sách phải vừa sức (ít mô hình, ưu tiên
  dùng checkpoint có sẵn, ít phải tự cài/tự train).
- **Q-c (Mamba so với gì):** thầy nói "có thể so sánh Mamba với nhiều thứ" —
  **người dùng chưa rõ ý thầy**. Cần tự đề xuất rồi xác nhận ở buổi Tuần 9.
- **Q-f (đề cương):** **sẽ phải sửa đề cương.** Luật repo vẫn giữ: không sửa
  file `docs/de_cuong/*.docx` đã nộp — đề cương điều chỉnh làm **bản mới**
  (file mới), bản cũ giữ làm lưu trữ.
