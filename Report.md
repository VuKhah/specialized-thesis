# Báo cáo các vấn đề cần thầy giải đáp

**Đề tài:** So sánh Mamba (SSM) và Conformer làm encoder trong ASR tiếng Việt
hội thoại (CTC, train từ đầu trên VietSuperSpeech)
**GVHD:** thầy Hoàng Văn Dũng · 
**Ngày lập:** 2026-09-26

## Tóm tắt tiến độ

- Pipeline đầy đủ (log-mel → encoder → CTC, tokenizer BPE 1.000, training
  loop có checkpoint/resume, đo WER) đã chạy với dữ liệu thật.
- Hai encoder đã khớp tham số: Mamba 12.292.352 vs Conformer 12.204.288
  (chênh **0,72 %**). Cả hai đã chạy thử train + eval trên Kaggle T4.
- Chưa train chính thức: đang chuẩn bị lưu trữ dữ liệu (~28 GB) trên Kaggle.

---

## A. Cần thầy quyết định (ảnh hưởng thiết kế thí nghiệm)

### A1. RQ2 — dữ liệu thật không đủ dải độ dài audio 

**Vấn đề.** Số liệu đo thật khác đề cương đã đăng ký:

| | Đề cương | Đo thật (HF Hub) |
|---|---|---|
| Số mẫu / số giờ | 52.023 / 267,39 h | **67.405 / 245,42 h** |
| Độ dài audio | 3–30 s | **10–15 s** (gần như đồng nhất) |

Dataset không thay đổi sau khi viết đề cương. Với dải 10–15 s, RQ2 (RTF theo độ dài, ưu thế O(n) của Mamba so
với O(n²) của Conformer) khó chứng minh thuyết phục.

**Các phương án:**
1. Giữ nguyên, đo RTF theo bucket hẹp 10–15 s.
2. Ghép các đoạn liên tiếp cùng video thành audio dài hơn, **chỉ để đo RTF** (không dùng để train).
**Đề xuất: pa 2.** Đo RTF không cần nhãn, nên dùng được ~17–21 nghìn chuỗi ≥ 3 đoạn liên tiếp, tạo dải 30–60 s trở lên. Không đổi dữ liệu train,
giữ nguyên câu hỏi nghiên cứu. (Nếu cần đo WER trên audio dài: chỉ có 57 chuỗi thuần tập validation.)

Rủi ro: Vì các Đoạn audio không phải được cắt liên tiếp, nên phương án ghép chưa được chốt.

### A2. Tập validation/test không độc lập với tập train

**Vấn đề.** 561/562 video của `validation` cũng có trong `train`;
6.748/6.749 câu validation (và toàn bộ 250 câu clean-test) đến từ video đã
gặp khi train. Không trùng file audio, nhưng WER đo được sẽ **lạc quan** hơn
thực tế (người nói/ngữ cảnh đã gặp).

**Phương án:** (a) giữ split của tác giả dataset; (b) rút ~5 % video khỏi
train để làm tập test "video chưa gặp" (đổi tập train so với đề cương).

Chưa có Đề xuất

### A3. Hạn chế kiến trúc 

- **Không subsampling** (chuỗi ~1.000–1.500 khung):   subsampling rút ngắn chuỗi, làm yếu chính RQ2. 

điều này làm chuỗi quá dài và tải mỗi epoch rất lâu. và vẫn chưa có giải pháp.

### A4. Chi phí train so với hạn mức GPU 

Full dataset hơn 27GB

Ước lượng hiện tại ~70 phút/epoch → 30 epoch ≈ **40 GPU-giờ/mô hình**, vượt hạn mức Kaggle 30 GPU-giờ/tuần.

**Đề xuất: giữ 30 epoch như đề cương.** Tăng tốc theo thứ tự rủi ro:
DataLoader song song → mixed precision (AMP) → 2 GPU, áp dụng giống hệt cho cả
hai mô hình. Mỗi mô hình train trên một tài khoản Kaggle riêng (của 2 người
khác nhau), chạy song song. Chỉ khi vẫn vượt quota mới xin thầy cho giảm số
epoch.

---

## B. Cần thầy xác nhận (hình thức khóa luận)

| # | Câu hỏi |
|---|---|
| B3 | Chương 2 "Phân tích và thiết kế": đề tài thiên về nghiên cứu, nên trình bày dạng sơ đồ kiến trúc/luồng dữ liệu hay cần UML như đề tài ứng dụng? |

---

