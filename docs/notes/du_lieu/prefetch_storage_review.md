# Review dataset VietSuperSpeech để quyết định Q2 (lưu audio prefetch ở đâu)

Ngày: 2026-09-26. Phục vụ quyết định ⏸ "Prefetch audio đầy đủ" (`TODO.md`,
`Plan.md` mục 5, `QA.md` Q2). **Tài liệu này chỉ phân tích, không chọn phương
án** — quyết định thuộc về người dùng.

Ký hiệu độ tin cậy: **[đo]** = đo trực tiếp phiên này bằng HF API / file thật ·
**[nguồn]** = tài liệu chính thức · **[cộng đồng]** = diễn đàn/blog, chưa kiểm
chứng · **[ước]** = suy ra từ số khác.

## 1. Tóm tắt

- Cần đúng **67.405 file WAV = 28,27 GB (26,33 GiB)** [đo]. Con số "~27 GB"
  cũ trong `TODO.md` thực ra là GiB.
- Nén **FLAC** (lossless, bit-exact) còn **~60,5 %** → **~17,1 GB** [đo trên
  300 file] — vừa `/kaggle/working` 20 GB, nhưng biên chỉ ~3 GB.
- Không phương án nào cần sửa kiến trúc. Khác nhau ở: tốn bao nhiêu GPU-giờ
  mỗi phiên, làm một lần mất bao lâu, và có phải sửa code dùng chung
  (đường dẫn cache / đuôi `.flac`) hay không.
- **Phát hiện phụ quan trọng (ảnh hưởng Q1/Q4, không phải Q2):** toàn bộ
  train+validation chỉ đến từ **1 kênh (vietcetera)**, không phải 4 kênh như
  bài báo/đề cương. Xem mục 3.

## 2. Số liệu dataset (đo 2026-09-26)

Repo HF `thanhnew2001/VietSuperSpeech`, commit cuối `cbf624ae9b`
(2026-02-22, không đổi từ đó — 1.323 commit, đều trong 19-22/02/2026) [đo].
Không gated, license khai báo trên card: **MIT** [nguồn].

| Hạng mục | Giá trị |
|---|---|
| Toàn repo | 62,62 GB, 118.264 file |
| Cần dùng (train+validation) | **67.405 file, 28,27 GB** (60.656 train / 6.749 dev) |
| Tổng thời lượng | 245,42 h (train 220,84 h, dev 24,57 h) |
| Định dạng | WAV PCM_16, 16 kHz, mono [đo 300 file] |
| Kích thước/file | min 320 KB · trung vị 429 KB · max 480 KB (= 10,0-15,0 s) |
| FLAC (300 file mẫu) | 126,0 MB → 76,2 MB, **tỉ lệ 0,605**, giải nén bit-exact |
| Đọc 1 file (SSD local) | WAV 0,56 ms · FLAC 4,55 ms |
| Đã có ở local | 14.641 file, 6,13 GB (`data/raw/audio_cache`, ~22 %) |

Phân bố file cần dùng theo thư mục HF (dùng khi phải chia gói < 20 GB):

| Thư mục | File cần | GB |
|---|---|---|
| `asr_segments_vietcetera_part00` | 8.980 | 3,76 |
| `…_part01` | 8.973 | 3,75 |
| `…_part06` | 8.951 | 3,75 |
| `…_part07` | 8.996 | 3,77 |
| `…_part08` | 8.971 | 3,78 |
| `…_part09` | 8.973 | 3,77 |
| `…_part13` | 8.987 | 3,78 |
| `…_part14` | 4.574 | 1,92 |

Ví dụ chia 2 gói WAV: {00, 01, 06, 07} = 15,03 GB · {08, 09, 13, 14} = 13,25 GB.

Phần **không dùng** (34,3 GB): `asr_dataset_nguoiviethaingoai` 18,95 GB
(11.119 file, **dài tới ~8.989 s ≈ 2,5 h/file** — audio thô, không có
transcript trong train/dev.json), `nguyenkhangofficial` 5,59 GB,
`trinhlieu` 4,86 GB, `vietcetera_part12/15` + phần dư part14 ~4,7 GB,
`nguoivietdailynews` 0,09 GB.

## 3. Ba phiên bản số liệu không khớp nhau

| Nguồn | Mẫu | Giờ | Nguồn audio | Độ dài đoạn |
|---|---|---|---|---|
| Bài báo arXiv 2603.01894 (02/03/2026) = đề cương | 52.023 | 267,39 | 4 kênh YouTube | 3-30 s |
| README card trên HF (cũ) | 32.267 | 103,18 | nguoivietdailynews, nguyenkhangofficial, trinhlieu | ~12 s TB |
| `manifest.json` + `train/dev.json` (**dữ liệu thật**) | 67.405 | 245,42 | **chỉ vietcetera** (dù manifest liệt kê 4 nguồn) | 10-15 s |

`manifest.json` ghi chú *"Only includes audio files that exist on HuggingFace
repo"* — giải thích hợp lý nhất: 3 nguồn còn lại (`nguoitrongmuonnghe`,
`nguoiviethaingoai`, `vietsuccess`) không có đoạn đã cắt trên repo nên bị loại
khỏi split [ước]. Hệ quả cần đưa vào `dataset_discrepancy.md` / hỏi GVHD
(**không xử lý ở đây**): độ đa dạng người nói/miền giảm so với mô tả trong đề
cương; thư mục `nguoiviethaingoai` có audio dài hàng giờ nhưng không có nhãn.

Nên **pin `revision="cbf624ae9b…"`** khi gọi `load_dataset`/`hf_hub_download`
để kết quả tái lập được nếu tác giả đẩy bản mới (hiện code không pin) —
thay đổi nhỏ ở code dùng chung, cần người dùng đồng ý.

## 4. Ràng buộc nền tảng

| Ràng buộc | Giá trị | Độ tin cậy |
|---|---|---|
| `/kaggle/working` | 20 GB, lưu lại khi Save Version | người dùng xác nhận 2026-09-21 + [cộng đồng] |
| Thư mục khác (vd. tự tạo `/kaggle/tmp`) | ~60 GiB trống, **mất khi hết phiên** | [cộng đồng — nhân viên Kaggle trả lời trên forum], chưa đo |
| `/kaggle/input` (dataset gắn vào) | chỉ đọc, **không tính vào 20 GB** | [nguồn/cộng đồng] |
| Dataset private | quota 200 GB/tài khoản; 1 dataset tối đa 200 GB | [cộng đồng — thông báo Kaggle "Doubling of Private Quota"], chưa mở được trang gốc |
| Output notebook | báo cáo giới hạn 500 file hiển thị/tải qua API; cách lách là nén tar/zip | [cộng đồng], chưa kiểm |
| Zip upload lên Dataset | được Kaggle tự giải nén | [cộng đồng], chưa rõ với dataset tạo từ output notebook |
| Phiên | GPU 9 h, CPU 12 h; GPU 30 h/tuần, reset thứ Bảy 00:00 UTC; **phiên CPU không tốn quota GPU** | [cộng đồng] |
| HF rate limit (resolver, cửa sổ 5 phút) | ẩn danh **3.000/IP** · tài khoản free **5.000** · PRO 12.000 | [nguồn — docs HF 09/2025] |

**Hệ quả của rate limit:** 67.405 lần `hf_hub_download` ⇒ tối thiểu **~112
phút** nếu ẩn danh, **~67 phút** nếu có `HF_TOKEN` — chỉ riêng giới hạn, chưa
tính băng thông [ước]. Ẩn danh trên Kaggle còn có thể **dùng chung IP** với
người khác → dễ dính 429. `prefetch_audio.py` hiện không truyền token. Tốc độ
thực trước đây (~1-4 s/file, 8 luồng) → 2,3-9,4 h cho toàn bộ [ước, đo ở
local, chưa đo trên Kaggle].

## 5. Các phương án cho Q2

Giả định chung: train trên Kaggle GPU; ~80 phút/epoch (Q3 chưa quyết).

| # | Phương án | Làm một lần | Chi phí mỗi phiên GPU | Sửa code dùng chung | Rủi ro / chưa kiểm |
|---|---|---|---|---|---|
| A | **Tải lại vào thư mục tạm mỗi phiên GPU** (`/kaggle/tmp`) | Không | **~1-9 h GPU-giờ/phiên** để tải (≥ 67-112 phút theo rate limit) — trên phiên 9 h mất 12-100 % | `AUDIO_CACHE_DIR` cấu hình được | Dung lượng ~60 GiB chưa đo; 429 từ HF; ngốn quota vốn đã thiếu (Q3) |
| B | **Kaggle Dataset WAV, tạo từ 2 notebook CPU** (mỗi notebook ~13-15 GB vào `/kaggle/working`, nén tar theo thư mục) | 2 phiên CPU (miễn phí GPU) | ~0 (gắn vào `/kaggle/input`) | Đọc từ 2 gốc (hoặc symlink về 1 cây) + đường dẫn cấu hình được | Giới hạn 500 file/tar; tar có được tự giải nén không; tốc độ đọc `/kaggle/input` |
| C | **Kaggle Dataset FLAC, 1 notebook CPU** (~17,1 GB) | 1 phiên CPU + thời gian nén | ~0 | Như B + đổi `.wav`→`.flac` khi mở file (soundfile đọc FLAC trong suốt; front-end không đổi vì lossless) | Biên 20 GB chỉ ~3 GB (ước từ 300 file); đọc chậm hơn ~4 ms/file — với `num_workers=0` hiện tại ≈ +70 ms/step batch 16 (~5 % trên 1,3 s/step) |
| D | **Tải về local rồi `kaggle datasets create` từ máy** (đã có 22 %) | Tải thêm ~22 GB + **upload 28 GB** từ mạng nhà | ~0 | Đường dẫn cấu hình được | Phụ thuộc băng thông upload của người dùng; không tốn giờ Kaggle nào |
| E | **Không lưu audio — tính trước log-mel** | — | — | **Có**, đụng front-end dùng chung | ~200 KB/mẫu (80×~1.250×fp16) ≈ 13 GB, không nhỏ hơn FLAC bao nhiêu; khóa cứng front-end, khó thêm augmentation. Nêu cho đủ, không khuyến nghị cân nhắc trước |

Ghi chú:
- Mọi phương án ngoài A đều cần **1 thay đổi chung cho cả hai encoder**:
  `AUDIO_CACHE_DIR` đọc từ config/env thay vì hằng số cứng
  (`src/data/vietsuperspeech_dataset.py:30`). Không làm lệch hai thí nghiệm.
- B và C tránh được việc mỗi phiên GPU phụ thuộc HF (mạng, rate limit, repo
  bị xoá/đổi) — cũng là một bản sao cố định cho tái lập kết quả.
- Dataset tạo trên Kaggle nên để **private**: license MIT là của người đăng
  HF, còn nội dung gốc là video YouTube của bên thứ ba.

## 6. Còn phải kiểm chứng trước khi chốt (đề xuất, chưa chạy)

Một kernel **CPU** ngắn (không tốn quota GPU) có thể trả lời hết:
1. `df -h` — dung lượng thật của `/kaggle/working`, `/kaggle/tmp`, `/tmp`.
2. Tải 1.000 file bằng `prefetch_audio` (có/không `HF_TOKEN` qua Kaggle
   Secrets) → tốc độ thật, có dính 429 không.
3. Nén 1.000 file sang FLAC → tỉ lệ trên mẫu lớn hơn (chốt biên 20 GB của C).
4. Ghi 1 tar > 500 file vào output → xem có tạo dataset được và có tự giải
   nén không (quyết định cách đóng gói B/C).

## 7. Chia 3 shard + train/val/test (bổ sung 2026-09-26, đề xuất — chưa chốt)

Bối cảnh: người dùng có **2 tài khoản Kaggle**; hướng đang bàn là mỗi tài
khoản train 1 mô hình (không gộp trọng số), dữ liệu train chia **3 Kaggle
Dataset** (shard) để mỗi notebook tạo dataset vừa 20 GB.

### Chia phân tầng — mô phỏng trên `train.json` thật, seed 42

Cách chia: nhóm theo `source` (video, 645 video), trong mỗi video sắp theo
`duration` rồi chia vòng tròn 0-1-2 (điểm bắt đầu xoay giữa các video để số
mẫu đều). Mỗi video vì vậy rải đều ~1/3 sang mỗi shard.

| | Mẫu | Giờ | GB WAV | Video có mặt | Ký tự phủ | TB giây | TB từ | %10s/11/12/13/14/15s |
|---|---|---|---|---|---|---|---|---|
| Toàn train | 60.656 | 220,84 | 25,44 | 645 | 96/96 | 13,107 | 45,88 | 11,9/13,9/16,2/20,7/23,9/13,3 |
| shard0 | 20.219 | 73,63 | 8,48 | 630 | 95/96 | 13,109 | 45,88 | 11,8/14,0/16,2/20,7/23,9/13,4 |
| shard1 | 20.219 | 73,60 | 8,48 | 634 | 96/96 | 13,105 | 45,92 | 11,9/13,9/16,3/20,7/23,9/13,3 |
| shard2 | 20.218 | 73,61 | 8,48 | 628 | 96/96 | 13,107 | 45,85 | 11,9/14,0/16,2/20,7/23,9/13,3 |

Video thiếu ở vài shard là các video chỉ có 1-2 câu (min 1, trung vị 93, max
590 câu/video) — không tránh được và không đáng kể. Nên kiểm thêm độ phủ
token BPE khi làm thật.

**Cách dùng shard khi train:** nên gắn cả 3 dataset cùng lúc (`/kaggle/input`
không tính 20 GB) và xáo trộn toàn cục — shard chỉ là cách lưu, việc train
không đổi. Nếu buộc phải lần lượt: 1 epoch = shard0→1→2 mỗi shard **1 lượt**,
thứ tự shard xáo mỗi epoch; **không** train hội tụ trên 1 shard rồi mới sang
shard khác (quên dần, lệch thứ tự). Hai tài khoản dùng **cùng manifest shard**.

### Phát hiện: validation không độc lập với train (theo video)

- 561/562 video của `validation` cũng có trong train; **6.748/6.749** câu
  validation đến từ video đã thấy khi train (không trùng file audio; 23 câu
  trùng y hệt transcript).
- Clean-test 250 câu lấy từ validation: cả 186 video đều có trong train.
- Hệ quả: WER validation/clean-test đo khả năng nhận dạng **người nói/video
  đã gặp** → lạc quan so với thực tế. So sánh Mamba vs Conformer vẫn công
  bằng (cùng điều kiện), nhưng con số tuyệt đối phải diễn giải đúng.
- Muốn có test "video chưa gặp" thì phải rút một số video khỏi train (vd.
  ~5 % video) — đổi tập train so với đề cương → hỏi GVHD (`QA.md` Q10).

### Chia shard và phương án 2 của RQ2 (ghép đoạn liên tiếp) — đo 2026-09-26

Phương án 2 trong `dataset_discrepancy.md` (ghép đoạn liên tiếp cùng video
thành audio dài) **vẫn đang chờ GVHD** (🔴), chưa chốt. Số liệu để cân nhắc:

- Tên file có chỉ số `_segNNN` theo thứ tự trong video. 67.405 đoạn có mặt /
  71.901 chỉ số → **6,3 % chỉ số bị thiếu** (lọc bỏ). Không có timestamp, nên
  hai chỉ số liền nhau cũng chưa chắc liền nhau về thời gian.
- Train/dev của tác giả chia **ngẫu nhiên theo từng đoạn**: chuỗi dev liên
  tiếp dài 1 đoạn: 5.542, 2 đoạn: 515, 3: 49, 4: 6, 6: 1.
- Số chuỗi ≥ 3 đoạn liên tiếp (~≥ 30 s): **chỉ dev: 57** · chỉ train: 17.585 ·
  lẫn train+dev: 21.795.
- Hệ quả: đo **RTF** (chỉ tốc độ, không cần nhãn) dùng chuỗi nào cũng được →
  không liên quan cách chia shard. Đo **WER trên audio dài** thì chỉ được dùng
  chuỗi thuần dev (57 chuỗi), nếu không là lọt dữ liệu train vào test.
- Chia shard vòng tròn trong mỗi video làm đứt chuỗi liên tiếp *trong một
  shard*, nhưng shard chỉ là cách lưu: gắn đủ 3 shard thì vẫn dựng chuỗi từ
  manifest đầy đủ. Nếu muốn giữ chuỗi trong từng shard: chia mỗi video thành
  3 **khối liên tiếp** thay vì vòng tròn (cân bằng theo video vẫn giữ).
- Trùng lặp/lọt giữa các shard: shard là **phân hoạch** của `train.json` —
  script tạo shard phải kiểm: 3 shard rời nhau, hợp lại đúng 60.656 file,
  không chứa file nào của dev/clean-test.

### Đề xuất train / val / test

| Tập | Nội dung | Dùng để | Lưu |
|---|---|---|---|
| Train | HF `train` 60.656 câu, 3 shard như trên | train | 3 dataset × 8,48 GB |
| Val | HF `validation` **trừ 250 câu clean-test** = 6.499 câu | eval định kỳ, chọn `best.pt` | 1 dataset ~2,8 GB, gắn mọi phiên |
| Test | clean-test 250 câu (sau khi hiệu đính tay) | báo cáo kết quả cuối, 1 lần | nằm trong dataset val (không dùng để chọn mô hình) |
| (tuỳ GVHD) Test video chưa gặp | vài % video rút khỏi train | đánh giá tổng quát hoá | — |

Trừ clean-test khỏi val sửa luôn rủi ro đã ghi ở `TODO.md` (`best.pt` chọn
theo validation chứa clean-test). Đây là sửa code dùng chung (danh sách câu
eval), áp dụng như nhau cho cả hai encoder.

## Nguồn

- HF dataset: https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech
- Bài báo: https://arxiv.org/abs/2603.01894
- HF rate limits: https://huggingface.co/docs/hub/en/rate-limits
- Kaggle disk (forum): https://www.kaggle.com/discussions/product-feedback/372506 ,
  https://www.kaggle.com/general/136779 , https://www.kaggle.com/product-feedback/542425
- Kaggle quota dataset: https://www.kaggle.com/product-announcements/512322
- Giới hạn 500 file output: https://www.kaggle.com/product-feedback/181143 ,
  https://github.com/Kaggle/kaggle-cli/issues/665
- Quota GPU: https://www.kaggle.com/docs/efficient-gpu-usage ,
  https://www.kaggle.com/product-feedback/173129
- Tổng hợp Kaggle (bên thứ ba): https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/kaggle_20251121.md
