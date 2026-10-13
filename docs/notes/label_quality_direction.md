# Hướng mới: chất lượng nhãn VietSuperSpeech và ảnh hưởng tới đánh giá

Ghi 2026-10-13 tối. **Người dùng chốt thêm hướng này vào khóa luận** sau khi thấy đề tài "chỉ khảo sát" còn đơn
điệu và muốn có phần "của mình". Chưa hỏi GVHD — mang trang này đi hỏi thầy Dũng (cùng đề cương mới).
Ràng buộc người dùng đặt: **không thêm run train**. Độ sâu đến từ khai thác dữ liệu đã có.

## 1. Điểm xuất phát

Transcript VietSuperSpeech là **nhãn giả**, sinh tự động bằng Zipformer-30M-RNNT-6000h qua Sherpa-ONNX,
**không người kiểm tay** (`docs/notes/de-tai-phowhisper-vietsuperspeech.md`; *cần kiểm lại thẻ dữ liệu trên HF để
trích nguồn chính xác*). Hệ quả:
- Mọi mô hình train từ đầu của dự án học từ nhãn của một mô hình khác.
- WER trên val / val_unseen đo so với chính nhãn máy đó.
- Clean-test 203 câu đang được hiệu đính tay (`src/data/correct_clean_test.py`) → nhãn thật duy nhất của dự án.

## 2. Bốn câu hỏi (đều không cần GPU train)

| # | Câu hỏi | Cách đo | Dữ liệu cần |
|---|---|---|---|
| L1 | Nhãn huấn luyện sai bao nhiêu? | WER(nhãn giả, nhãn hiệu đính) = WER của Zipformer trên clean-test; tách theo loại lỗi (thanh điệu, dấu phụ, tên riêng/tiếng Anh, xóa/chèn) | bộ hiệu đính |
| L2 | Học trò có vượt thầy không? | WER từng mô hình so với nhãn hiệu đính, đặt cạnh WER của Zipformer (L1); bootstrap ghép cặp theo video | `predictions.tsv` + bộ hiệu đính |
| L3 | Mô hình có học lại lỗi của nhãn không? | Tại mỗi vị trí nhãn giả sai: mô hình sai giống hệt nhãn ("nhại") / sửa đúng / sai khác; so tỉ lệ "nhại" giữa Mamba B1, ConExt, Conformer, Parakeet | căn chỉnh 3 chuỗi (nhãn giả, nhãn thật, dự đoán) |
| L4 | Đánh giá bằng nhãn giả lệch bao nhiêu? | WER trên nhãn giả so với trên nhãn thật cho từng mô hình; thứ hạng có đổi không | như L2 |

Nhánh kết luận viết sẵn (theo tinh thần quy tắc 9.4):
- L2: mô hình tốt hơn Zipformer (CI không chứa 0) → "lọc được một phần nhiễu nhãn"; kém hơn → "nhãn giả là trần";
  CI chứa 0 → "không phân biệt được".
- L4: thứ hạng đổi → cảnh báo cho mọi nghiên cứu dùng VietSuperSpeech chỉ đo trên nhãn giả; không đổi → "nhãn giả
  đủ để xếp hạng nhưng sai độ lớn WER" (nếu độ lệch đáng kể).

## 3. Vì sao chọn hướng này

- Là câu hỏi mới về bộ dữ liệu, không phải thêm một bảng so sánh mô hình.
- Gần 0 GPU-h: dùng dự đoán đã có (`reports/results/final_compare_2026-10-08/predictions.tsv`, sau này dự đoán
  checkpoint cuối của bảng chính) + bộ hiệu đính.
- Biến đường găng (hiệu đính 203 câu) thành dữ liệu chính của một phần đóng góp.
- Nối vào RQ3: tách "lỗi thanh do nhãn" và "lỗi thanh do mô hình".

## 4. Giới hạn phải khai

- 203 câu, 29 video → mọi số kèm CI bootstrap theo video.
- **Neo nhãn:** ô hiệu đính điền sẵn nhãn giả → người hiệu đính có thể giữ lỗi của nhãn → L1 bị đánh giá thấp.
  Giảm nhẹ (người dùng quyết): gõ lại từ đầu một mẫu nhỏ câu (vd. 20) không xem nhãn giả, đo độ lệch giữa hai cách.
- Một người hiệu đính, không đo độ đồng thuận giữa người.
- Clean-test là video giữ riêng → kết luận về chất lượng nhãn chỉ đại diện cho phân bố đó.

## 5. Việc cần làm (chưa làm gì ở phiên 13/10)

1. Kiểm thẻ dữ liệu HF: nguồn chính thức về cách sinh nhãn (mô hình, phiên bản).
2. Hiệu đính đủ 203 câu (đang 11/203) — người dùng.
3. Script phân tích L1-L4 (mở rộng `src/evaluation/error_analysis.py`: căn chỉnh 3 chuỗi), test trên 11 câu đã có.
4. Thêm vào đề cương mới + mục lục (`docs/khoa_luan/muc_luc_de_xuat.md`): RQ riêng hay gộp RQ3 — **người dùng/GVHD chọn**.
5. Dashboard mô phỏng (https://claude.ai/artifact/BjsgBGGfZyamnsw9RBXx8y): thêm tab "Nhận xét" + phần chất lượng nhãn.

## 6. Bối cảnh quyết định trong phiên 13/10 (để khai đúng thứ tự)

- Seed 2 (~50 GPU-h): người dùng không ưu tiên vì chỉ kiểm độ ổn định, không làm mô hình mạnh hơn.
- "Train dài hơn ở 40 ms" (~33 GPU-h, ước 80 epoch): người dùng **thiên về**, chưa chốt số epoch → chưa có config, chưa đẩy.
- Đặc trưng cao độ (pitch) cho thanh điệu: AI đề xuất rồi **người dùng bác** (thêm run + thêm bảng, tốn thời gian).
  Nguồn đã tra: Ghahremani và cs. ICASSP 2014 (chỉ đọc abstract), Interspeech 2023 huu23 (phát hiện phát âm sai, không
  phải ASR), arXiv 2602.10003. Giữ làm "hướng phát triển".
- Người dùng: "về độ sâu của nhận xét thì thật sự phải có" → ưu tiên nhận xét từ dữ liệu sẵn có (độ phân giải × kiến
  trúc, chi phí theo độ dài, lỗi thanh điệu, WER theo GPU-giờ) + hướng chất lượng nhãn ở trên.
