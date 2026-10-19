# Hướng dẫn hiệu đính clean-test

Cập nhật 2026-10-14 (bản đầu 2026-10-08). Công cụ: `src/data/correct_clean_test.py`.

## 0. Vì sao việc này quan trọng nhất lúc này

Nhãn của VietSuperSpeech do mô hình Zipformer-30M sinh tự động, không người kiểm. Clean-test 203 câu
(29 video giữ riêng, 44,3 phút audio) là **tập duy nhất có nhãn người** của khóa luận. Mọi kết luận độ chính
xác (RQ1), phân tích lỗi (RQ3) và hướng chất lượng nhãn (L1–L4, `docs/notes/du_lieu/label_quality_direction.md`) đều
chấm trên bộ này, **một lần**, sau khi xong đủ 203 câu. Biên bản phản biện 14/10
(`docs/notes/review/phan_bien_2026-10-14.md`) xếp đây là đường găng số 1.

Tiến độ 14/10: **11/203** (cả 11 câu thuộc 2 video). Ước công sức: 3–5 phút làm cho mỗi phút audio →
**khoảng 2,5–4 giờ**, chia nhiều buổi được.

## 1. Bật công cụ

Mở terminal ở **gốc repo** (`D:\workspaces\specialized-thesis`), rồi chạy:

```
python -m src.data.correct_clean_test
```

- Mở trình duyệt tại **http://127.0.0.1:7861** (cổng 7860 để cho demo).
- Dòng in ra lúc khởi động phải có `thiếu audio: 0`. Đã kiểm 14/10: đủ 203 file trong `data/raw/audio_cache/`.
- Khởi động mất khoảng 10–20 giây vì phải nạp model Conformer để tô chữ theo audio (mục 2). Chờ đến dòng
  `tô chữ theo audio: ...` rồi mở trình duyệt.
- Cổng 7861 đang bận thì chạy `python -m src.data.correct_clean_test --port 7862`.
- Lỗi font chữ Việt trên console Windows: đặt `PYTHONIOENCODING=utf-8` trước lệnh (PowerShell:
  `$env:PYTHONIOENCODING="utf-8"`).
- Tắt công cụ: `Ctrl+C` ở terminal. Không mất gì, vì mỗi câu được ghi ngay khi bấm Lưu.

**An toàn dữ liệu:** mỗi lần mở, công cụ tự sao lưu manifest thành
`data/processed/clean_test_manifest.bak-<thời gian>.json`. Kết quả ghi vào
`data/processed/clean_test_manifest.json` (cột `corrected_text`, `notes`). Đây là file duy nhất trong
`data/processed/` được track trên git, nên **commit sau mỗi buổi làm** để khỏi mất.

## 2. Màn hình và các nút

| Thành phần | Dùng để |
|---|---|
| Dòng đầu | Số câu, trạng thái (✅/⬜), tiến độ `x/203`, video, thời lượng |
| Audio | Tự phát khi mở câu; nghe lại bằng nút play, kéo thanh để nghe đoạn khó (tốc độ phát: nút ⋮ của trình phát) |
| **Theo dõi lời** | Chữ của *Bản hiệu đính* tô vàng theo audio đang phát; **bấm vào một chữ để nghe từ chữ đó**. Cập nhật ngay khi sửa ô |
| Nhãn giả (Zipformer) | Chỉ để tham khảo, không sửa được |
| **Cố vấn: Whisper large-v3** | Bản chép của một model **ngoài đội hình** (mục 2a). Chỉ tham khảo |
| **Bản hiệu đính** | Ô để sửa. Câu chưa làm được điền sẵn nhãn giả (chữ thường) |
| Khung so sánh | Chỗ khác với nhãn giả: đỏ gạch = bỏ, xanh = thêm, kèm số từ khác |
| Ghi chú | Nhãn đặc biệt cho câu (mục 4) |
| **💾 Lưu & tiếp** | Ghi câu này rồi nhảy tới câu **chưa làm** kế tiếp |
| ◀ Trước / Bỏ qua ▶ | Đi lùi / tới mà không lưu |
| Đến câu số + Đi | Nhảy tới một câu |
| Đến câu chưa làm | Nhảy tới câu chưa làm gần nhất |
| Xoá hiệu đính câu này | Đưa câu về trạng thái chưa làm |

Công cụ **từ chối lưu** khi ô trống hoặc còn chữ số.

**Tô chữ hoạt động thế nào:** công cụ căn (forced alignment CTC) đúng chữ trong ô hiệu đính với audio, dùng
Conformer v1 (`checkpoints/avg_v1/conformer_avg20-29.pt`) chạy CPU, độ phân giải 10 ms. Model chỉ cho biết *khi
nào*, không gợi ý chữ nào. Vị trí tô là ước lượng, có thể sớm/muộn một chút. Khi bạn gõ một chữ không có trong
audio, model vẫn phải nhét nó vào đâu đó: nếu vài chữ liền nhau bị tô vụt qua cùng lúc, hoặc một chữ đứng
rất lâu, thì nên nghe lại đoạn đó. Không có checkpoint thì công cụ vẫn chạy, chỉ không tô.

## 2a. Cố vấn Whisper (từ 14/10 tối)

Người hiệu đính vẫn là người quyết; model chỉ là ý kiến thứ hai. Công cụ hiện thêm:

- Ô **Cố vấn: Whisper large-v3**: bản chép của Whisper (OpenAI), chạy một lần trên Kaggle cho đủ 203 câu
  (`scripts/kaggle/whisper_advisor/`, output local `data/processed/whisper_advisor.json`, không track git).
- Khung **Đối chiếu**: mỗi chữ trong bản hiệu đính được tô theo việc hai model ngoài đội hình (Zipformer và
  Whisper) có cùng chữ đó không. Không tô: cả hai có. **Cam**: một bên có. **Đỏ**: không bên nào có, nên nghe
  lại. Bên dưới là phần khác biệt so với Whisper.

Mỗi model mạnh ở một chỗ (thử 14/10 trên 7 câu khó):

| | Zipformer (nhãn giả) | Whisper |
|---|---|---|
| Lặp từ, từ đệm, nói vấp | Giữ khá đủ | **Hay bỏ** (*một một* → *một*) |
| Tên công ty, tiếng Anh | Phiên âm sai (*câu trách*, *teaming point*) | **Thường đúng** (*Gojek*, *tipping point*, *P&G*) |
| Số | Viết bằng chữ | **Ra chữ số** (*700*) → chữ đỏ oan, cứ viết bằng chữ |
| Lỗi riêng | Nghe nhầm âm gần | Đôi khi **bịa/lặp** cả cụm |

Vì sao được xem chữ của Whisper mà không được xem chữ của các mô hình đang so sánh: nhãn sửa theo một model
thì model đó (và model giống nó) được lợi khi chấm. Whisper không nằm trong đội hình nên không thiên vị bên nào
trong phép so sánh. Tiền lệ: Faria et al. 2022 (arXiv 2206.06192, mục 2.1) khi audit nhãn Switchboard cho chuyên
gia tham khảo bản chép độc lập nhưng không cho xem hệ ASR đang được đánh giá. Ngoại lệ phải khai: **PhoWhisper**
(dự bị) fine-tune từ Whisper.

**Tên riêng/tên công ty không biết viết thế nào:** tên file là tiêu đề tập Vietcetera, và trang của tập ghi khách
mời, công ty, host (vd. tập OPES: host Miro Nguyễn → chữ *miro* là tên host). Cách viết chuẩn đã xác minh nằm ở
`docs/notes/du_lieu/clean_test_entities.md` (có link nguồn). Tên không xác minh được: ghi theo âm nghe được, ghi chú
`tên chưa xác minh`.

### Output các mô hình của đề tài (từ 14/10, 23:17)

Người dùng chọn xem cả chữ của 5 mô hình đang so sánh (Mamba B1, ConExtBiMamba, Conformer-12M ở checkpoint ep28;
Parakeet và PhoWhisper zero-shot; lấy từ `reports/results/final_compare_2026-10-08/predictions.tsv`). Các bản chép
này nằm trong khung gập **Mô hình của đề tài** và được so với bản hiệu đính.

- **Rủi ro, ghi vào phần Hạn chế:** nhãn sửa theo chữ của một mô hình thì mô hình đó được lợi khi chấm. Vì vậy:
  tự nghe và sửa trước, chỉ mở khung này để đối chiếu chỗ còn phân vân.
- Từ mốc này, mỗi lần bấm Lưu thì câu được ghi thêm `corrected_at`. 113 câu làm trước mốc không có trường này.
  Lúc chấm, có thể báo thêm WER chỉ trên các câu làm trước mốc (phân tích độ nhạy).
- Chữ của các mô hình train từ đầu đôi khi dính từ (*vấnuby*). Đó là output thật của mô hình, không phải lỗi hiển thị.

## 3. Quy tắc ghi

Nguyên tắc chung: **ghi đúng những gì người nói thật sự nói**, không ghi theo cái Zipformer nghe ra.

1. **Nguyên văn từng từ**, kể cả lặp từ, nói vấp, từ đệm (*thì, là, mà, á, à, ừ, ờ, hả, ạ, nha, nhé*) nếu
   nghe rõ. Không sửa ngữ pháp, không làm câu "đẹp" hơn.
2. **Dấu câu, hoa/thường: không quan trọng.** WER được tính sau khi chuyển về chữ thường và bỏ dấu câu.
3. **Số viết bằng chữ**, theo đúng cách người nói đọc: *hai mươi bảy*, *năm hai nghìn không trăm hai mươi*,
   *mười lăm phần trăm*.
4. **Kiểu bỏ dấu thanh: theo kiểu cũ, giống nhãn train**, tức *hóa, hòa, lý, kỹ, mỹ*, không viết *hoá, hoà, lí,
   kĩ, mĩ*. Đã kiểm 14/10: nhãn giả của 203 câu dùng kiểu cũ cho tất cả các trường hợp này. Giữ nhất quán
   để WER không phạt oan vì khác kiểu bỏ dấu, trong khi quy tắc chuẩn hóa hóa/hoá vẫn chưa chốt.
5. **Tiếng Anh chen giữa câu**: viết đúng chính tả tiếng Anh (*marketing, team, clip, livestream, new york*).
   Từ viết tắt đọc theo chữ cái thì viết liền chữ thường (*nft, dna, mc*). Thêm ghi chú `tiếng Anh`.
6. **Tên riêng** (người, địa danh, thương hiệu): viết đúng chính tả thật, kể cả khi nhãn giả viết sai. Thêm
   ghi chú `tên riêng`.
7. **Từ bị cắt nửa ở đầu hoặc cuối đoạn audio**: bỏ, chỉ ghi các từ nghe trọn.
8. **Đoạn khó nghe** (nhạc nền, nhiều người nói chồng, nói nhỏ): ghi phỏng đoán tốt nhất, thêm ghi chú
   `khó nghe` (hoặc `nhạc nền`, `chồng tiếng`). Không bỏ câu.
9. **Không phải tiếng Việt** (sót sau bước lọc ngôn ngữ): ghi chú `không phải tiếng Việt`, vẫn lưu. Việc loại
   hay giữ câu đó do bạn/GVHD quyết sau, không tự xoá.
10. **Câu nhãn giả đã đúng hoàn toàn**: vẫn bấm **Lưu & tiếp**. Câu đó tính là đã hiệu đính, và việc nhãn
    đúng cũng là dữ liệu cho câu hỏi L1.

## 4. Ghi chú: viết đúng từ khóa

Script phân tích lỗi đọc ô Ghi chú bằng **từ khóa**, không phân biệt hoa/thường. Nhiều nhãn cách nhau bằng
dấu phẩy, ví dụ `tên riêng, khó nghe`.

| Ghi | Dùng cho |
|---|---|
| `tên riêng` | Câu có tên người, địa danh, thương hiệu. Được tách thành nhóm riêng khi tính WER |
| `tiếng Anh` | Câu có từ tiếng Anh. Cùng nhóm tách riêng |
| `khó nghe`, `nhạc nền`, `chồng tiếng` | Chất lượng audio kém |
| `không phải tiếng Việt` | Câu sót ngôn ngữ khác |

## 5. Những lỗi dễ mắc

- **Bị neo theo nhãn giả.** Ô được điền sẵn nhãn của Zipformer, nên rất dễ "nghe thấy" đúng như chữ đã viết.
  Cách giảm: **nghe audio trước, đọc nhãn sau**. Với chỗ nghi ngờ, nghe lại đoạn đó ít nhất hai lần rồi mới
  quyết. Đây là một hạn chế sẽ khai trong khóa luận.
- **Bỏ sót từ đệm** ở đầu câu (*thì, à, ừ*), thường bị Zipformer nuốt mất.
- **Sửa theo nghĩa thay vì theo âm**, ví dụ người nói "cái này nó" mà viết thành "cái này". Ghi đúng âm đã nói.
- **Tin tô chữ hơn tai.** Tô chữ chỉ giúp định vị; chữ được tô đúng lúc không có nghĩa là chữ đó đúng.
- **Gõ chữ số** (*27*, *2020*). Công cụ sẽ chặn; sửa thành chữ.
- **Gõ sai bộ gõ**: dấu thanh đặt lệch, hoặc ký tự tổ hợp. Công cụ đã chuẩn hoá NFC, nhưng nên dùng bảng mã
  Unicode dựng sẵn.

## 6. Nhịp làm gợi ý

- Mỗi buổi 30–45 phút (khoảng 40–60 câu), rồi nghỉ. Nghe lâu dễ bị neo nhãn hơn.
- Làm **theo thứ tự công cụ đưa ra**, không chọn câu dễ trước. 192 câu còn lại trải trên 27 video chưa đụng tới.
- Cuối mỗi buổi: `git add data/processed/clean_test_manifest.json` rồi commit. File này chỉ chứa transcript
  của bộ dữ liệu công khai, đưa lên repo public được.

## 7. Khi xong đủ 203 câu

1. Báo mình. Không cần tự chạy eval.
2. Chốt quy tắc chuẩn hóa văn bản (`normalize_text`, việc hóa/hoá…) **trước** khi chấm. Đây là quyết định của
   bạn, đang nằm trong TODO.
3. Mình chạy một kernel eval duy nhất cho mọi checkpoint đã có trên 203 câu nhãn người (ước dưới 1 GPU-h, cần
   bạn duyệt), rồi tính CI bootstrap theo video, các câu hỏi L1–L4 và phân tích lỗi theo âm tiết.
4. Theo quy tắc 9.4 mục 3, sau lần chấm này không quay lại sửa nhãn. Muốn sửa câu nào thì làm **trước** bước 3.

## 8. Phương pháp (để ghi vào Chương 2–3 và phần Hạn chế)

- Một người hiệu đính, không đo độ đồng thuận giữa nhiều người.
- Ô được điền sẵn nhãn giả, nên có rủi ro neo nhãn. Hướng giảm (mục 5) được ghi lại.
- Không điền sẵn output của các mô hình đang so sánh, để không thiên vị mô hình nào.
- Tô chữ theo audio (từ 14/10, sau 11 câu đầu) dùng Conformer v1 để căn thời gian cho văn bản người hiệu đính
  tự viết. Model không đưa ra chữ nào, nên không phải nguồn gợi ý nội dung.
- Video giữ riêng không đồng nghĩa với người nói chưa gặp (MC có thể xuất hiện ở video khác).
