# Tiền xử lý audio — đối chiếu tài liệu và số đo dữ liệu (2026-10-05)

Bổ sung cho `frontend_decoder_survey.md` (đã chốt log-mel 80, CMVN toàn cục,
SpecAugment ngày 2026-10-04). File đó xử lý **từ log-mel trở đi**; file này xét
phần **trước** log-mel (waveform) và các phép tăng cường trên waveform. Lý do
xem lại: train thử 2026-10-05, câu B1 ra rỗng/NaN đi theo độ to của audio
(Plan.md nhật ký 2026-10-05 chiều).

**Trạng thái: đề xuất, chưa áp dụng.** Front-end là code dùng chung của nhóm
train từ đầu → đổi gì cũng áp cho cả 3 encoder và cần người dùng duyệt.

## 1. Pipeline hiện tại (đọc code)

`vietsuperspeech_dataset.__getitem__`: `sf.read` → float32, **không** kiểm
sample rate/số kênh, **không** chuẩn hoá biên độ, không dither, không
pre-emphasis, không tăng cường trên waveform. Sau đó `log_mel.py`: MelSpectrogram
(n_fft 400, hop 160) → `log(clamp 1e-5)` → CMVN toàn cục → SpecAugment (train).

## 2. Số đo dữ liệu (local, 2026-10-05)

| Đo | Kết quả |
|---|---|
| Định dạng (958 wav val_unseen có ở local) | 100% 16 kHz, mono, PCM_16 — không cần resample/downmix, chỉ nên **kiểm** (assert) |
| RMS dBFS, 2000 câu train ngẫu nhiên (seed 0) | p1 −33,8 · p5 −28,3 · p50 −20,5 · p95 −14,8 · p99 −12,2 |
| RMS trung bình theo video (136 video ≥ 5 câu) | p5 −28,3 · p50 −20,5 · p95 −16,1 → chênh lệch chủ yếu **giữa các video** (kênh/thiết bị thu) |
| Câu có đỉnh ≥ 0,999 (chạm trần, có thể clip từ nguồn) | 5,1% |
| DC offset | không đáng kể (\|mean\| ≤ 0,0015) |

Hệ quả toán học (không cần nguồn): nhân waveform với hệ số g thì phổ công suất
nhân g², log-mel **cộng hằng số 2·ln g ở mọi khung, mọi kênh** (khi chưa chạm
`clamp`). p5 → p95 độ to ≈ 13,5 dB ↔ lệch ≈ 3,1 nat, xấp xỉ 1 độ lệch chuẩn
log-mel của `cmvn_stats.json` (std theo kênh 2,8–3,7). CMVN **toàn cục** trừ
một mean cố định nên **không** khử được lệch này; CMN **theo câu** khử chính xác.

Liên hệ B1 (val_unseen, eval epoch 4): tỉ lệ câu rỗng theo video tăng theo độ
to — video ≈ −30 dB: 10%, ≈ −25 dB: 13%, ≈ −17 đến −21 dB: 40–100%.
**Kernel `diag-b1-asr` bác giả thuyết "to → NaN":** nhân gain ×0,25…×4 cho kết
quả không đơn điệu (nhóm hỏng: 12, 15, 20, 15, 5 /20; nhóm lành: 2, 3, 0, 3, 3
/20), tương quan RMS ↔ đỉnh kích hoạt chỉ 0,26. NaN là tràn fp16 bên trong khối
Mamba (Plan.md nhật ký 2026-10-05 tối), độ to chỉ là một trong các yếu tố đầu
vào. Điều kernel cho thấy: đầu ra B1 đổi mạnh khi chỉ đổi gain → mô hình chưa bất
biến với độ to — lý do phụ cho đề xuất 2, **không** phải cách sửa NaN.

## 3. Tài liệu: các bước tiền xử lý trên waveform

| Kỹ thuật | Bằng chứng công bố | Hệ đối chứng dùng | Ghi chú cho dự án |
|---|---|---|---|
| **Speed perturbation** 0,9/1,0/1,1 | Ko và cs., Interspeech 2015: 4 tác vụ LVCSR 100–960 h, **cải thiện tương đối trung bình 4,3%**; khuyến nghị chính của bài | ConMamba yaml `speed_changes: [95, 100, 105]`; torchaudio có `SpeedPerturbation` | Chỉ lúc train, trên waveform → RQ2 không đổi. Tốn CPU resample + câu dài/ngắn hơn ±10% (s/step đổi). Là tăng cường có bằng chứng định lượng rõ nhất |
| **Volume (gain) perturbation** | Kaldi `perturb_data_dir_volume.sh`: gain ngẫu nhiên [0,125; 2], "typically useful … when using systems that don't have cepstral mean normalization" — thực hành bộ công cụ, **không tìm được ablation công bố** | Kaldi recipe nnet3/chain | Đúng trường hợp dự án (CMVN toàn cục, không CMN theo câu). Rẻ (nhân 1 số) |
| **CMN/CMVN theo câu** | Kaldi `apply-cmvn` mặc định theo câu/người nói; một so sánh (kết quả tìm kiếm, chưa đọc bản gốc) thấy global ≈ theo câu | NeMo FastConformer `normalize: per_feature` | Khử gain chính xác nhưng **đã bị loại 2026-10-04** vì RQ2 (audio ghép dài nhận phép biến đổi khác lúc train). Nếu muốn: chỉ trừ mean theo câu (không chia std) — vẫn phụ thuộc độ dài, cần bàn lại với RQ2 |
| **Chuẩn hoá biên độ waveform** (peak/RMS theo câu) | Không tìm được bài so sánh cho E2E ASR | Whisper chuẩn hoá ở mức log-mel (toàn cục), không ở waveform | Tương đương CMN theo câu về mặt gain; RMS theo câu phụ thuộc nội dung (im lặng nhiều → khuếch đại nhiễu) |
| **Dither** | — | NeMo `dither: 0.00001` | Chỉ tác dụng ở đoạn im lặng tuyệt đối (log của 0); dự án đã `clamp(1e-5)` → lợi ích không rõ |
| **Pre-emphasis** 0,97 | — | Kaldi, NeMo mặc định | Không có bằng chứng cho log-mel + mạng sâu; bỏ qua |
| **Thêm nhiễu/RIR** (MUSAN…) | Ko 2015 cũng xét; cần dữ liệu nhiễu ngoài | icefall, ESPnet tuỳ recipe | VietSuperSpeech đã là audio thật nhiều nhiễu; tốn công, ngoài phạm vi |

## 4. Đề xuất (người dùng chọn — chưa làm)

1. **Kiểm định dạng** trong `__getitem__`: assert sr = 16000, mono. Không đổi số
   liệu (dữ liệu đã đúng), chỉ chặn lỗi im lặng. *Không cần duyệt phương pháp.*
2. **Volume perturbation lúc train** (gain ngẫu nhiên, vd. ±6 dB hoặc theo Kaldi
   [0,125; 2]) — nhắm thẳng vào chênh lệch độ to giữa video (mục 2), giữ CMVN toàn
   cục và RQ2. Bằng chứng: thực hành Kaldi, không có ablation.
3. **Speed perturbation** 0,9/1,0/1,1 (Ko 2015) hoặc 0,95/1/1,05 (ConMamba) —
   bằng chứng tốt nhất, nhưng đổi s/step và cần ước lại GPU-giờ train full.
4. Không đề xuất dither, pre-emphasis, nhiễu ngoài (không có bằng chứng/ngoài phạm vi).

Áp cho cả 3 mô hình train từ đầu (Conformer, ConExtBiMamba vừa train ổn vẫn phải
dùng cùng front-end → phải train thử lại cả 3 nếu đổi, hoặc chỉ đổi từ train full).

## 5. Tra thêm (2026-10-05, người dùng yêu cầu trước khi chốt)

**Speed perturbation — đọc lại Ko 2015 (bản gốc):** hệ **DNN-HMM lai**, không phải
E2E, **không có SpecAugment**; 0,9/1,1 tạo bản sao **offline ×3**. Bảng 3: GALE
Mandarin 100 h −2,0% · Tedlium 118 h −3,9% · Switchboard 300 h −6,7% · LibriSpeech
960 h −3,2% (TB 4,3%) · ASpIRE 5.500 h chỉ −0,3% → lợi giảm khi dữ liệu nhiều; 176 h
của dự án nằm trong dải có lợi. Speed tốt hơn tempo perturbation và VTLP (Bảng 1).

**SP khi đã có SpecAugment (E2E):** thực hành phổ biến dùng cả hai — ESPnet
Conformer (Guo và cs., arXiv:2010.13956: đa số tập dùng cả hai; † chỉ SP, ‡ chỉ
SpecAugment theo từng tập, **không phải ablation có kiểm soát**), icefall Zipformer
(0,9/1,0/1,1), ConMamba (0,95/1/1,05). **Không tìm được ablation có kiểm soát "SP
cộng thêm trên SpecAugment" cho E2E train từ đầu.** EURASIP JASMP 2026
(doi 10.1186/s13636-026-00451-8, low-resource) dùng SP + SpecAugment làm mốc mạnh
và thấy cộng thêm FadeOutIn tốt nhất — chỉ đọc được tóm tắt, chưa đọc bản đầy đủ.

**On-the-fly vs ×3:** không tìm được bài so sánh. ×3 nhân 3 chi phí mỗi epoch →
không khả thi với quota (B1 đã 18,1 GPU-h/30 epoch). On-the-fly (mỗi câu chọn ngẫu
nhiên 1 hệ số mỗi lần đọc) giữ nguyên số step/epoch; độ dài TB đổi ~+0,7%
(1/0,9 và 1/1,1 không đối xứng); câu dài nhất 15 s → 16,7 s.

**Volume perturbation:** vẫn chỉ có thực hành Kaldi; Liu và cs. (SLTU-CCURL 2020,
arXiv:1909.06522) theo tóm tắt kết quả tìm kiếm thấy "additional volume
perturbation was not helpful" ở thí nghiệm đơn ngữ (chưa đọc bản gốc). Cộng với
`diag-b1-asr` bác "to → NaN" → bằng chứng yếu.

**CMVN toàn cục vs theo câu:** một so sánh (arXiv:2011.04884, SLU độ trễ thấp)
thấy "almost the same performance", toàn cục tốt hơn khi xử lý theo đoạn → giữ
toàn cục (đã chốt vì RQ2).

**Đề xuất cập nhật:** (1) assert định dạng — làm; (2) SP on-the-fly 0,9/1,0/1,1
cho cả 3 mô hình **nếu** preflight cho thấy DataLoader không thành nút cổ chai
(resample CPU); không làm ×3; (3) không làm volume perturbation; (4) giữ CMVN toàn
cục. Hệ quả của (2): kết quả train thử (không SP) không còn cùng điều kiện với
train full — chỉ dùng để loại phương án, đã đúng mục đích.

## Nguồn

- Ko, Peddinti, Povey, Khudanpur — Audio augmentation for speech recognition, Interspeech 2015: https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html
- torchaudio `SpeedPerturbation`: https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.SpeedPerturbation.html
- Kaldi `perturb_data_dir_volume.sh`: https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/data/perturb_data_dir_volume.sh
- Kaldi `apply-cmvn`: https://kaldi-asr.org/doc/apply-cmvn_8cc.html
- NeMo FastConformer CTC config: https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml
- ConMamba yaml (speed perturb, bf16, global norm): https://github.com/xi-j/Mamba-ASR/blob/main/hparams/CTC/conmamba_large.yaml
- Guo và cs., Recent Developments on ESPnet Toolkit Boosted by Conformer: https://arxiv.org/abs/2010.13956
- EURASIP JASMP 2026, SP + SpecAugment low-resource: https://link.springer.com/article/10.1186/s13636-026-00451-8
- Liu và cs., Multilingual Graphemic Hybrid ASR with Massive Data Augmentation: https://arxiv.org/abs/1909.06522
- Global vs utterance CMVN (SLU độ trễ thấp): https://arxiv.org/abs/2011.04884
- mamba-ssm README (AMP giữ tham số fp32; "SSMs are sensitive to their recurrent dynamics"): https://github.com/state-spaces/mamba
