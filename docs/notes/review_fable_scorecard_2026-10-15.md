# Chấm điểm khóa luận và "nếu là tôi thì làm gì" (Fable 5.1, 2026-10-15)

Bổ sung cho `review_fable_2026-10-15.md`. Đây là **ý kiến cá nhân của một người phản biện**, không phải điểm của hội
đồng. Thang 100 chia theo cấu trúc rubric Mẫu 2 mà repo ghi (`docs/khoa_luan/muc_luc_de_xuat.md`: 40 điểm cho tính
đúng đắn của phương pháp/thiết kế, 20 điểm hỏi đáp); phần còn lại tôi chia theo thông lệ chấm KLTN nghiên cứu.
Chấm hai cột: **(A) nộp đúng những gì đang có hôm nay** (giả định viết trung thực, không thêm số) và **(B) làm đúng
phương án khuyến nghị** (vá 2 lỗ hổng, 3 seed ở 40 ms, viết đủ, bảo vệ tốt).

## 1. Bảng điểm

| Hạng mục (trọng số) | A: hôm nay | B: sau phương án | Căn cứ |
|---|---|---|---|
| **Phương pháp và thiết kế thực nghiệm (40)** | 29 | 35 | Thiết kế có kiểm soát, tham số khớp, test theo video, bootstrap khối, quy tắc chốt trước: mức tốt của KLTN. Trừ vì: bảng 10 ms thiếu Conformer relPE và dùng sai checkpoint (ep28 thay vì ep37); 1 seed; không dò LR; Conformer không PE là baseline phi chuẩn; "40 ms" trộn 4 biến. B vá được 3 điểm đầu, hai điểm sau chỉ khai |
| **Kết quả và phân tích (20)** | 12 | 17 | Có kết quả thật, có CI, có RQ2 đủ đường cong. Trừ vì: kết luận 40 ms chưa tách khỏi nhiễu seed; RQ3 mới có đếm từ dính và 11 câu phân tích lỗi cũ; L1–L4 chưa làm; chưa có bảng cuối thống nhất |
| **Tính mới / giá trị (10)** | 6 | 8 | Lần đầu đo Mamba cho ASR tiếng Việt hội thoại ở hai độ phân giải (theo phạm vi tra cứu đã khai); kết luận "không mặc nhiên tốt hơn" + ngưỡng độ dài là hữu ích. Không phải đóng góp phương pháp mới; dữ liệu nhãn giả giới hạn tầm khái quát |
| **Bản viết (15)** | 5 | 11 | Hôm nay: Ch1 nháp chưa đối chiếu trích dẫn, Ch2/Ch3 mới là note, Mở đầu/Kết luận trống, đề cương mới chưa duyệt. Đây là chỗ mất điểm nặng nhất và cũng là chỗ dễ lấy lại nhất nếu bắt đầu viết Ch3 ngay khi có bảng |
| **Sản phẩm phần mềm, tái lập (5)** | 4 | 5 | Pipeline chạy thật, kernel ghim commit, metrics đầy đủ, demo CPU. Còn: demo chưa có 3 mô hình 40 ms; repo chưa commit nhiều file; notebook nhúng audio |
| **Hỏi đáp trước hội đồng (10, quy đổi từ 20)** | 6 | 8 | Nếu hỏi "vì sao Mamba thua Conformer", "nhãn giả tin được không", "1 seed thì nói gì": hiện trả lời được nhưng phải dựa vào biện luận; sau B có số để trả lời |
| **Tổng** | **62** | **84** | |

Giải thích con số:
- **62 hôm nay** không phải vì thí nghiệm yếu, mà vì **quyển chưa tồn tại** và bảng chính còn hai lỗi quy trình. Một hội
  đồng đọc đúng những gì repo có sẽ thấy phương pháp tốt hơn trung bình nhưng sản phẩm cuối chưa đủ để chấm cao.
- **84 sau phương án** là mức tôi tin đạt được nếu làm đúng, không cần thêm ý tưởng nào. Trần của đề tài với dữ liệu
  và tài nguyên này tôi ước **~88**: muốn hơn phải có nhiều seed ở cả 10 ms, dò LR, hoặc tập test lớn hơn với nhiều
  người hiệu đính — những thứ đã loại vì thời gian.
- Rủi ro kéo xuống dưới 75 chỉ có một: **viết không kịp**. Mọi thứ khác đã có đường xử lý.

## 2. Điều tôi đánh giá cao nhất và điều tôi lo nhất

**Cao nhất:** kỷ luật phương pháp. Quy tắc 9.4 viết trước khi có số, khai cả sự cố AI thấy WER relPE sớm, không giấu
việc ConExt thua B1, bootstrap theo khối video. Nếu bảo vệ, phần này nên được nói to, vì 40% điểm nằm ở đó.

**Lo nhất:** không phải GPU, mà là ba thứ của con người: (1) đề cương mới chưa duyệt và đã quá 2 tuần chưa gặp GVHD
(Mẫu 3 ghi hậu quả rõ); (2) Ch2–Ch3 chưa có chữ nào ở dạng quyển; (3) thói quen đổi hướng theo từng kết quả mới —
chính người dùng đã nhận ra điều này ngày 13/10. Phương án khuyến nghị cố ý nhỏ để chặn (3).

## 3. Nếu là tôi, tôi làm gì (theo thứ tự, có mốc)

**72 giờ tới (16–18/10), không chờ gì cả:**
1. Sáng 16/10: tải output run Conformer relPE 10 ms (B s3) và 3 checkpoint ep37 10 ms. Nếu output B còn, lỗ hổng 1 đóng
   ngay. Nếu mất, quyết định luôn: chạy lại 18 GPU-h trên B từ 17/10 (xong ~20/10) — đây là trường hợp duy nhất tôi
   chấp nhận train lại ở 10 ms.
2. Cùng ngày: hoàn thiện đề cương mới theo RQ ở review mục 2.1 (một buổi), chuyển Word, **gửi GVHD kèm xin lịch
   gặp tuần 19–25/10**. Đính kèm bảng clean-test đã có. Không chờ số seed để gửi.
3. Chốt 12 quyết định ở review mục 4 bằng một dòng mỗi mục trong `Plan.md` mục 4, ghi ngày. Viết tiêu chí 3 seed
   (review 2.3) vào `conextbimamba_gap_2026-10-12.md` thành mục 9.6 **trước** khi đẩy kernel.
4. 17/10 sau reset quota: kernel dump log-prob (4 checkpoint 10 ms + Parakeet fine-tune), rồi đẩy 6 run seed: A ConExt
   ×2, B relPE ×2, C B1 ×2. Mỗi tài khoản 2 kernel song song (C đã làm được). Kiểm 10 phút đầu mỗi kernel.
5. Trong lúc chờ: dựng **khung Ch3 trong Word ngay hôm nay** với mọi bảng để trống nhưng đã có tên, cột, chú thích,
   câu dẫn. Khi số về chỉ điền. Đây là việc tôi coi quan trọng ngang kernel.

**Tuần 19–25/10:**
6. 20–21/10: chấm một lần đủ 9 run 40 ms + 4 checkpoint 10 ms + Parakeet FT; giải mã beam + LM CPU; bootstrap; vẽ
   hình. Khoá bảng. Từ đây không mở file kết quả để "xem thử" nữa.
7. Gặp GVHD: trình đề cương, Bảng 1–2 có CI, hình RQ2, lịch viết. Hỏi đúng 3 câu: duyệt RQ; dạng "thiết kế" Ch2; hình
   thức nộp. Không hỏi "có nên làm thêm X không".
8. Gõ lại mù 20 câu (1 giờ). 25/10 đóng băng GPU, sao lưu checkpoint ep37 mọi run lên Drive ngay.

**26/10–15/11:** viết Ch3 trước (số liệu còn nóng), rồi Ch2 từ `ch2_nhap.md`, rồi sửa Ch1 theo bảng kiểm trích dẫn.
Mỗi tuần một buổi gặp hoặc gửi GVHD một chương. RQ3 và L1–L4 làm xen, giới hạn 3 ngày; hết 3 ngày thì dùng những gì
có.

**16/11–06/12:** Ch4 demo (thêm 3 mô hình 40 ms, Conformer relPE chạy CPU), Mở đầu, Kết luận, TLTK đối chiếu nguồn,
bản đủ 27/11, slide, chạy thử demo, in, CD.

## 4. Ba câu tôi sẽ chuẩn bị sẵn cho hội đồng

- *"Đề tài về Mamba mà Conformer lại tốt hơn?"* — Đúng ở 40 ms với 3 seed [điền số]; ở 10 ms thì ngược lại; Mamba thắng
  rõ ở audio dài (RQ2). Kết quả âm có kiểm soát là đóng góp; chọn trước quy tắc đọc kết quả nên không phải "kể chuyện".
- *"Nhãn train do máy sinh thì tin được không?"* — Đo được: nhãn giả lệch bản hiệu đính 1,8% WER, 0,65% trên từ Việt,
  37% trên từ ngoại; thứ hạng mô hình không đổi giữa hai loại nhãn; mọi WER tuyệt đối báo trên nhãn hiệu đính.
- *"Một seed thì khác biệt 0,5 điểm có ý nghĩa không?"* — Vì thế nhóm 40 ms chạy 3 seed; nhóm 10 ms 1 seed nhưng chênh
  3–9 điểm; độ lệch seed đo được là [điền số].
