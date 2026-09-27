# QA — câu hỏi đang mở

Danh sách ngắn các câu hỏi **đang chờ trả lời** (người dùng / GVHD), để mang
đi hỏi hoặc tự quyết. Mỗi mục chỉ tóm 1-2 dòng và trỏ tới nơi có chi tiết —
trạng thái đầy đủ vẫn ở [`TODO.md`](TODO.md), quyết định ở
[`Plan.md`](Plan.md) mục 4-5. Trả lời xong thì ghi quyết định vào `Plan.md`
và xoá mục khỏi file này.

Cập nhật: 2026-09-26. Bản tổng hợp các câu hỏi cho GVHD (Q1, Q3, Q5, Q6,
Q9, Q10) để mang đi gặp thầy: [`Report.md`](Report.md) (2026-09-27).

## Chặn tiến độ

| # | Câu hỏi | Ai trả lời | Chi tiết |
|---|---|---|---|
| Q1 | RQ2: audio thật chỉ 10-15 s (đề cương 3-30 s) → chọn phương án nào trong 4 phương án? Đề xuất AI (D12): phương án 2, ghép đoạn liên tiếp chỉ để đo RTF | GVHD | `docs/notes/dataset_discrepancy.md`, `training_plan_kaggle.md` D12 |
| Q2 | Đã chốt hướng: 3 Kaggle Dataset + 2 tài khoản + chạy nền. Còn duyệt chi tiết D1-D10 (WAV/FLAC, cách chia shard, val trừ clean-test, checkpoint theo step, chạy kernel kiểm chứng) | Người dùng | **`docs/notes/training_plan_kaggle.md`** mục 3; số liệu: `prefetch_storage_review.md` |
| Q3 | Train 30 epoch ≈ 40 GPU-giờ/mô hình > 30 GPU-giờ/tuần. Đề xuất D8/D9/D11: đo benchmark trước, rồi `num_workers` → AMP → DDP; giữ 30 epoch; **không subsampling (đã chốt)** | Người dùng (+ GVHD nếu giảm epoch) | `TODO.md` ⚠️ |

## Không chặn, cần xác nhận

| # | Câu hỏi | Ai trả lời | Chi tiết |
|---|---|---|---|
| Q4 | Đề cương đã nộp ghi 52.023 mẫu/267,39 h hay 67.405/245,42 h? (bản `.docx` trên đĩa khác bản từng commit) | Người dùng | `Plan.md` nhật ký 2026-09-21 |
| Q5 | Tên đề tài trên bìa: `Conformer` hay `Con-Former` (như đề cương)? | GVHD | `TODO.md` 📘 |
| Q6 | Hình thức nộp hiện hành (biểu mẫu 2018 còn dùng?) và dạng "thiết kế" mong đợi ở Chương 2? | GVHD | `Plan.md` mục 6 |
| Q7 | Lịch gặp GVHD: lần gần nhất / lần tới là ngày nào? | Người dùng | `Plan.md` mục 6 |
| Q8 | Có viết lại lịch sử git để gỡ `.docx` cũ, file TLCN có tên+MSSV người khác, audio nhúng trong notebook EDA khỏi repo public không? | Người dùng | `TODO.md` ⚠️ Repo public |
| Q9 | Hạn chế thiết kế cần nêu trong khóa luận hay sửa: Mamba đơn hướng + chưa mask padding; Conformer không có mã hóa vị trí. (Không subsampling: đã chốt giữ nguyên 2026-09-26, nêu là lựa chọn có chủ đích.) | Người dùng / GVHD | `ARCHITECTURE.md` mục 8 |
| Q10 | 99,9 % câu validation (và toàn bộ clean-test) đến từ video đã có trong train → WER lạc quan. Giữ nguyên và nêu hạn chế, hay rút ~5 % video khỏi train làm test "video chưa gặp"? Đề xuất AI (D13): giữ nguyên, nêu hạn chế | GVHD | `docs/notes/prefetch_storage_review.md` mục 7 |
