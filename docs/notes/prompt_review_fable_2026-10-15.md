# Prompt gọi Fable 5.1 rà soát toàn dự án (soạn 2026-10-15)

Cách dùng: mở phiên Claude Code mới **tại gốc repo**, chọn model Fable 5.1 (`/model`), dán nguyên khối dưới.

---

Bạn là người phản biện độc lập cho một khóa luận tốt nghiệp (KLTN) ngành CNTT về nhận dạng tiếng nói tiếng Việt.
Nhiệm vụ: **rà soát toàn bộ dự án và đề xuất MỘT phương án cuối cùng đúng đắn nhất về khoa học** cho phần còn lại
(thực nghiệm + cách viết kết luận). Người dùng và GVHD là người chốt; bạn đưa khuyến nghị rõ ràng, có căn cứ.
Trả lời bằng tiếng Việt.

## Yêu cầu cốt lõi của người dùng

1. **Ý nghĩa khóa luận phải thật.** Kết luận phải là phát hiện thật từ so sánh có kiểm soát, trả lời được câu hỏi
   có giá trị cho ASR tiếng Việt hội thoại. Không dựng câu chuyện đẹp, không chọn cấu hình có lợi cho Mamba, không
   kết luận vượt bằng chứng (CI chứa 0 → "không phân biệt được"). Kết quả bất lợi cho giả thuyết cũng phải báo.
2. **Phương án tốt nhất và đúng đắn nhất**, không chỉ "đủ để bảo vệ". Nếu cần thêm thực nghiệm GPU để kết luận
   đứng vững thì nói rõ, kèm GPU-giờ. Nếu thực nghiệm chỉ "đẹp thêm" mà không đổi kết luận thì nói là không cần.
3. **Ổn định:** người dùng đã mệt vì đổi hướng nhiều lần — phương án phải là bản chốt, không phải thêm một vòng thử.

## Đọc theo thứ tự (đừng giả định, kiểm trong repo)

1. `CLAUDE.md` — luật của repo (bắt buộc tuân thủ, xem mục "Giới hạn" bên dưới).
2. `TODO.md` các khối ▶ trên cùng; `Plan.md` mục 1–6 và **các mục cuối nhật ký mục 7** (từ 2026-10-12).
3. `ARCHITECTURE.md` (luồng dữ liệu, cạm bẫy mục 8).
4. Kết quả và ghi chú mới nhất:
   - `docs/notes/clean_test_results_2026-10-15.md` — **WER clean-test đã hiệu đính tay, quy tắc chấm, nhận xét**.
   - `docs/notes/glued_words_2026-10-15.md` — lỗi "từ dính" của CTC, giải mã beam + LM, phương án BPE3000 / CR-CTC
     (mục 3.2 có số liệu bài CR-CTC đã kiểm).
   - `docs/notes/conextbimamba_gap_2026-10-12.md` mục 9 — phạm vi "A + 40 ms", quy tắc phân tích chốt trước (9.4).
   - `docs/notes/label_quality_direction.md` — hướng chất lượng nhãn VietSuperSpeech.
   - `docs/notes/dataset_discrepancy.md`, `docs/notes/rq2_pilot_2026-10-05.md`, `reports/results/rq2_final_2026-10-13/`.
   - `docs/notes/de_cuong_moi_nhap.md` (nháp đề cương mới, chưa duyệt), `docs/notes/phan_bien_2026-10-14.md`.
   - `docs/notes/gvhd_buoi_1.md` (GVHD dặn "đừng ôm đồm"), `QA.md`.
5. `reports/results/` cho số liệu gốc; `git log --oneline -30`.

## Tóm tắt hiện trạng (kiểm lại với file, file là nguồn đúng)

- Dữ liệu: VietSuperSpeech (YouTube hội thoại, nhãn giả do Zipformer sinh), sau lọc ngôn ngữ train 45.442 câu
  (~175 h), đoạn 10–15 s. Test: clean-test 203 câu từ 29 video giữ riêng, **người dùng đã hiệu đính tay xong**
  (nhãn giả lệch bản hiệu đính 1,8% WER; 3% số từ là tiếng Anh/tên riêng/viết tắt).
- Nhóm train từ đầu (~12M tham số, chỉ encoder khác nhau, CTC, BPE 1000, 1 seed): Mamba hai chiều B1,
  ConExtBiMamba, Conformer. Hai cấu hình: v1 10 ms (không subsampling, 30 epoch) và 40 ms (subsampling 4×,
  38 epoch có pha giảm LR; Conformer dùng PE tương đối "relPE").
- Nhóm pre-train: Parakeet-CTC-0.6B-vi (zero-shot + fine-tune), PhoWhisper-small (zero-shot).
- WER clean-test (greedy): 40 ms relPE 0,121 · B1 0,126 · ConExt 0,131; v1 B1 0,170 · ConExt 0,202 · Conformer 0,265;
  Parakeet zero-shot 0,138 · PhoWhisper 0,149. Thứ hạng Mamba/Conformer **đảo** giữa v1 và 40 ms. Parakeet fine-tune
  chưa chấm clean-test.
- RQ: RQ1 độ chính xác, RQ2 hiệu quả theo độ dài audio (RTF/VRAM/độ trễ, audio ghép nhân tạo), RQ3 phân tích lỗi.

## Ràng buộc tài nguyên và thời gian

- Kaggle, mỗi tài khoản 2× T4, **30 GPU-giờ/tuần**, reset hằng tuần (lần tới 17/10). Hiện có 3 tài khoản (A, B, C);
  **người dùng có thể thêm, tối đa 10 tài khoản** (~300 GPU-giờ/tuần). Một phiên kernel ≤ ~9–12 h, nối phiên bằng
  checkpoint. Một run 40 ms hiện tốn 4,9–6,0 h.
- Mốc đã đặt: đóng băng GPU 01/11/2026 (có thể đề xuất đổi nếu có lý do khoa học); nộp GVHD tuần 30/11–06/12
  (`Plan.md` mục 2, 6). Gặp GVHD ≤ 2 tuần/lần; đề cương mới chưa duyệt.
- Phần viết khóa luận (4 chương) là đường găng — tính thời gian viết vào phương án.

## Câu hỏi cần trả lời

1. **Luận điểm chính** của khóa luận nên là gì để vừa thật vừa có giá trị? (vd. "Mamba vs Conformer ở quy mô nhỏ
   trên tiếng Việt hội thoại: phụ thuộc cấu hình thế nào", "đánh đổi WER–tốc độ theo độ dài", "chất lượng nhãn giả"…)
   Đề xuất cách phát biểu RQ cuối.
2. **Mối đe doạ lớn nhất tới tính đúng** của kết luận hiện tại, xếp hạng. Ít nhất xem: 1 seed (chênh 0,4–1 điểm giữa
   ba encoder 40 ms có thể nằm trong dao động seed); so sánh công bằng theo GPU-giờ (relPE đắt hơn); thứ hạng đảo
   v1 → 40 ms; nhãn giả cho train/val; 29 video test; RQ2 dùng audio ghép nhân tạo; zero-shot vs train trên cùng
   miền dữ liệu.
3. **Thực nghiệm còn thiếu — cần hay không:** nhiều seed (bao nhiêu, cho cấu hình nào); CR-CTC (đã có số liệu bài);
   BPE lớn hơn; chấm Parakeet fine-tune trên clean-test; trục GPU-giờ; RQ2 cuối. Với mỗi cái: phục vụ RQ nào, đổi
   kết luận ra sao, GPU-giờ, rủi ro. Tận dụng khả năng thêm tài khoản nếu thật sự cần, nhưng nhớ "đừng ôm đồm".
4. **Cách báo cáo:** bảng chính/phụ (greedy vs beam + LM), tách nhóm train từ đầu và pre-train, những gì phải khai
   ở mục hạn chế.
5. **Lịch cụ thể** từ nay đến nộp: việc GPU (tài khoản, ngày), việc viết, mốc gặp GVHD.

## Giới hạn khi làm việc (bắt buộc)

- **Chỉ đọc và phân tích.** Không chạy kernel Kaggle, không dùng GPU, không sửa code, không commit/push.
- Tuân thủ `CLAUDE.md`: không tự chốt các quyết định đang mở (bạn khuyến nghị, người dùng chốt); không sửa
  `docs/de_cuong/*.docx`; repo public — không đưa khoá, username Kaggle, tên người thứ ba vào file.
- Mọi đề xuất kỹ thuật phải có nguồn công bố đã kiểm (ghi link/arXiv); chỗ không có bằng chứng thì nói rõ là suy
  luận. Số liệu dự án phải trích từ file trong repo, không đoán.
- Chạy Python local: đặt `PYTHONIOENCODING=utf-8`, chạy từ gốc repo.

## Đầu ra

Ghi vào `docs/notes/review_fable_2026-10-15.md` (tạo mới), đồng thời tóm tắt trong chat:
1. Đánh giá hiện trạng: điểm mạnh, các mối đe doạ tính đúng (xếp hạng, mỗi mục có bằng chứng từ repo).
2. **Phương án khuyến nghị duy nhất**: luận điểm + RQ cuối; thực nghiệm làm / không làm (GPU-giờ, tài khoản, lịch);
   cách báo cáo; mục hạn chế phải khai.
3. Các phương án đã cân nhắc và loại, mỗi cái 1–2 dòng lý do.
4. Danh sách quyết định người dùng cần chốt, mỗi mục kèm khuyến nghị của bạn.
5. Rủi ro lịch trình và phương án dự phòng.
