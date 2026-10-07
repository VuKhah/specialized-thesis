# Hướng dẫn hiệu đính clean-test (2026-10-08)

Mục tiêu: thay nhãn giả (Zipformer sinh) của **203 câu clean-test** (29 video giữ
riêng, ~44 phút audio) bằng transcript do người nghe kiểm, để WER clean-test
trong khóa luận là WER so với nhãn đúng. Ước công sức: 3-5 phút nghe + sửa cho
mỗi phút audio → **khoảng 2,5-4 giờ**, chia nhiều buổi được (lưu từng câu).

## Công cụ

```
python -m src.data.correct_clean_test      # mở http://127.0.0.1:7861
```

- Chạy từ gốc repo. Audio đọc từ `data/raw/audio_cache/` (đã đủ 203 file).
- Mỗi lần mở công cụ, một bản sao lưu manifest được chép vào
  `data/processed/clean_test_manifest.bak-<thời gian>.json` (gitignore).
- Bấm **Lưu & tiếp** là ghi ngay vào `data/processed/clean_test_manifest.json`
  (cột `corrected_text`, `notes`), rồi nhảy tới câu chưa làm kế tiếp. Tắt giữa
  chừng không mất gì; mở lại thì công cụ vào thẳng câu chưa làm đầu tiên.
- Ô hiệu đính được điền sẵn nhãn giả (chữ thường) để sửa, không phải gõ lại.
  Phần khác biệt so với nhãn giả hiện ngay bên dưới (đỏ = bỏ, xanh = thêm).
- **Xoá hiệu đính câu này**: đưa câu về trạng thái chưa làm (eval dùng lại nhãn giả).

## Quy tắc ghi

Theo cách ghi của nhãn train VietSuperSpeech (đã xem nhãn giả của 203 câu) và
cách `normalize_text` chuẩn hoá trước khi tính WER.

1. **Ghi đúng từng từ được nói (nguyên văn)**, kể cả lặp từ, nói vấp, từ đệm
   (*thì, là, á, à, ừ, ờ, hả, ạ*) nếu nghe rõ — nhãn train có giữ từ đệm.
   Không sửa ngữ pháp, không "làm đẹp" câu.
2. **Không cần dấu câu, hoa/thường tuỳ ý** — WER tính sau khi đưa về chữ
   thường và bỏ dấu câu.
3. **Số viết bằng chữ**: *hai mươi bảy*, *năm hai nghìn không trăm hai mươi*.
   Nhãn không có chữ số và chuẩn hoá giữ nguyên chữ số → công cụ không cho lưu
   khi còn chữ số.
4. **Tiếng Anh chen giữa câu**: viết đúng chính tả tiếng Anh như nhãn train
   (*marketing, team, clip, new york*); từ viết tắt đọc theo chữ cái viết liền
   (*nft, dna*).
5. **Từ bị cắt nửa ở đầu/cuối đoạn**: bỏ, chỉ ghi từ nghe trọn.
6. **Khó nghe** (nhạc nền, chồng tiếng, nói nhỏ): ghi phỏng đoán tốt nhất và
   ghi chú `khó nghe` (hoặc `nhạc nền`, `chồng tiếng`) ở ô Ghi chú. Không bỏ câu.
7. **Không phải tiếng Việt** (sót sau lọc A1): ghi chú `không phải tiếng Việt`
   — người dùng/GVHD quyết loại hay giữ, không tự xoá câu.
8. Câu nhãn giả đã đúng: vẫn bấm **Lưu & tiếp** (tính là đã hiệu đính).

## Lưu ý phương pháp (ghi vào Ch3 / Hạn chế)

- Ô hiệu đính điền sẵn nhãn giả → có thể **neo** người nghe vào nhãn của
  Zipformer (một mô hình ngoài đội hình). Không điền sẵn output của các mô hình
  đang so sánh để tránh thiên vị cho mô hình nào.
- Một người hiệu đính, không đo độ đồng thuận giữa người nghe.
- Sau khi xong: chạy lại `python -m src.evaluation.eval_clean_test` cho các mô
  hình (script in số câu dùng `corrected_text`) và zero-shot; WER so với nhãn
  giả giữ lại làm đối chiếu.
