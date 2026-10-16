# Lỗi "từ dính" của mô hình CTC — chẩn đoán, phương án (2026-10-15, đang làm dở)

Người dùng phát hiện ConExt ra *nghỉơi, mìnhỉ, ngắmí*. Người dùng muốn "kết quả tốt nhất có thể", sẵn sàng train
lại/tiếp nếu đáng. Ba agent chạy song song: chẩn đoán, huấn luyện, giải mã. Dữ liệu và script của agent:
`D:\Model\kltn_decoding_2026-10-15\{diagnosis,training,decoding}\` (chép ra khỏi scratchpad của phiên, ngoài repo).

## 1. Chẩn đoán (agent, xong) — nguyên nhân là tokenizer, code không có bug

- BPE-1000 cắt âm tiết ngoài vocab thành `▁<phụ âm đầu>` + mảnh vần (`NGHỈ NGƠI → ▁NGH Ỉ ▁NG ƠI`). Ranh giới từ
  nằm trên phụ âm đầu, âm ngắn (đỉnh phụ âm → đỉnh vần trung vị 30 ms). CTC bỏ mảnh phụ âm (argmax = blank ở 85% ca
  mất) → vần dính vào từ trước. Token cả từ đúng (vd. `▁BẢO`) thường vẫn trong top 3 → greedy bỏ phí thông tin.
- Đếm trên 203 câu clean-test (predictions 08/10, ep28): Conformer 387 từ dính · ConExt 159 · Parakeet-CTC 137 · B1
  86 · PhoWhisper (AED) 5 → lỗi của CTC + đơn vị nhỏ hơn âm tiết, không riêng encoder nào.
- Không có bug: `greedy_collapse` (`src/models/ctc_model.py:103-116`), `BPETokenizer.decode` đúng; normalize(raw) ==
  hyp 203/203. Không tương quan tốc độ nói/vị trí/độ dài câu. Nhãn train gần như không có từ dính (60/2,36 triệu).

## 2. Huấn luyện (agent, xong) — phương án, chưa chọn

- Trên val_unseen ep37, nhãn giả: bản 40 ms từ dính 6,5–7,8/1000 từ, chỉ 12–14% số lỗi → sửa hết lợi ≤ 1,3–1,5 điểm.
- Tokenizer: BPE1000 13,0% nhãn đích là mảnh không ▁ · BPE2000 5,0% · **BPE3000 2,7%** (99% âm tiết có dấu = 1 mảnh,
  755 mảnh xuất hiện < 50 lần) · BPE5000 1,2% (3.050 mảnh hiếm).
- Phương án: (1) BPE~3000, rẽ nhánh 8 epoch từ `epochs/epoch29.pt` của 3 run v2a-s4 (head khởi tạo lại, AdamW mới),
  ~4,5 GPU-h cho 3 mô hình (+~1,5 h nhánh đối chứng head-reinit BPE1000) — cần code `--init_from --reinit_head` trong
  `train.py`, configs `v2b`, launcher; (2) CR-CTC (arXiv 2410.05101, ICLR 2025; LibriSpeech test-other Zipformer-M
  6,02 → 4,61), train từ đầu ~18–36 GPU-h; (3) self-conditioned CTC (2104.02724); (4) head phụ AED/transducer;
  (5) RNN-T/TDT. Không đề xuất: label smoothing, blank penalty, ký tự, distill Parakeet.
- Checkpoint ep29/ep37 của 3 run 40 ms **còn** trong output Kaggle (ConExt: A; relPE + B1: C); ep37 đã tải về
  `D:\Model\kltn_checkpoints\v2a_s4\`.

## 3. Giải mã (agent dừng theo phiên; bảng hoàn tất phiên 15/10 tối — mục 3.1)

Beam search CTC (flashlight trong torchaudio) + từ điển âm tiết + LM 3-gram Kneser-Ney tự viết. Chỉnh trên nửa
"tune" theo video của 958 câu val_unseen, báo trên nửa "dev" (517 câu). Clean-test **không** dùng.

- **Rò rỉ đã phát hiện và sửa:** `data/processed/train_transcripts.txt` (14/9, trước khi tách video giữ riêng) chứa
  nguyên văn 179/203 câu clean-test + câu val_unseen. LM sạch `lm3c` train từ `train_transcripts_filtered.txt`, loại
  câu trùng. Conformer v1 trên tune: 0,147 (LM bẩn) ≈ 0,146 (LM sạch) → rò rỉ không thổi phồng đáng kể, nhưng chỉ dùng
  LM sạch.
- Số sơ bộ đọc từ `final_dev.jsonl` (file không ghi tên mô hình; dòng greedy 0,2105 là Conformer v1 10 ms, dòng
  greedy ~0,102–0,103 là một mô hình 40 ms — **phải xác minh lại với báo cáo agent**):
  - Conformer v1, dev: greedy 0,2105 (từ dính 3,7%) · beam + từ điển 0,1865 · (trên tune: + LM 0,146)
  - Mô hình 40 ms, dev: greedy 0,1029 · beam + từ điển 0,102 · **beam + từ điển + LM 0,0889 (từ dính 0)**.
    Tham số LM weight 0,8, word score 1,0, beam 10; RTF giải mã ~0,025 CPU.
- Log-prob 3 mô hình 40 ms (ep37) trên 958 câu + clean-test: kernel `<A>/dump-logprobs-v2a-s4` (4 phút GPU),
  dataset vào `<A>/kltn-v2a-s4-ep37`, output `D:\Model\kltn_checkpoints\logprobs_v2a_s4\` (greedy trùng eval
  lúc train 943–951/958 câu). WER greedy subset: B1 0,0996 · ConExt 0,0983 · Conformer relPE 0,0961.

### 3.1 Bảng cuối — nửa dev val_unseen (517 câu, 14 video, 26.536 từ; nhãn giả; ep37, 40 ms)

Dòng `final_dev.jsonl` 16/20 được tạo trong phiên trước (lấy từ scratchpad cũ); 4 dòng beam 30 chạy phiên 15/10 tối
(`beam30_dev.jsonl`). Tất cả chép về `D:/Model/kltn_decoding_2026-10-15/decoding/`. Tham số LM dùng chung cho cả
ba: lmw 0,8 / ws 1,0 = điểm có WER **trung bình 3 mô hình** thấp nhất trên nửa tune (`sweep4.jsonl`: 0,0809; kế đó
0,8/0,5 = 0,0813, 0,5/0,5 = 0,0815) — không chỉnh riêng cho mô hình nào.

| WER (từ dính %) | greedy | greedy + tách từ dính (LM) | beam 10 + từ điển | **beam 10 + từ điển + LM** | beam 30 + từ điển + LM |
|---|---|---|---|---|---|
| B1 (Mamba) | 0,1029 (0,87) | 0,0991 | 0,1020 | **0,0889** (0) | 0,0869 |
| ConExtBiMamba | 0,1020 (0,94) | 0,0975 | 0,0978 | **0,0859** (0,01) | 0,0838 |
| Conformer relPE | 0,0992 (0,85) | 0,0959 | 0,1105 | **0,0812** (0,01) | 0,0791 |
| *Conformer v1 10 ms (tham khảo)* | 0,2105 (3,70) | 0,1862 | 0,1865 | 0,1577 | 0,1504 |

RTF giải mã CPU (một luồng / tiến trình, 4 tiến trình song song): beam 10 + LM ~0,022–0,026; beam 30 ~0,11–0,12.

Bootstrap theo video (`src/evaluation/bootstrap.py`, 10.000 lần; **chỉ 14 video** → khoảng tin cậy thô):
- Greedy: B1 − ConExt +0,0009 [−0,0012; +0,0070] (không phân biệt) · B1 − relPE +0,0037 [+0,0012; +0,0089] ·
  ConExt − relPE +0,0028 [−0,0003; +0,0045].
- Beam 10 + từ điển + LM: B1 − ConExt +0,0030 [+0,0010; +0,0064] · B1 − relPE +0,0076 [+0,0049; +0,0134] ·
  ConExt − relPE +0,0047 [+0,0013; +0,0085] → thứ hạng giữ nguyên, khoảng cách **giãn ra** sau LM.
- Lợi ích LM (final − greedy, cùng mô hình): B1 −1,40 [−2,04; −1,05] · ConExt −1,61 [−1,94; −1,27] · relPE −1,79
  [−2,48; −1,43] điểm.

Nhận xét (dữ liệu có sẵn, không thêm GPU):
- LM xoá gần hết từ dính (0,85–0,94% → ≤ 0,01%) và hạ WER 1,4–1,8 điểm cho cả ba; lợi nhiều nhất ở relPE.
- Beam + từ điển **không** LM giúp ConExt/B1 chút ít nhưng **làm relPE xấu đi** (0,0992 → 0,1105) — chưa rõ nguyên nhân
  (giả thuyết, chưa kiểm: phân bố log-prob relPE nhọn khác, beam 10 không LM bị từ điển cắt cụt ở đoạn tiếng Anh;
  dòng này chạy không có `--fallback`).
- Beam 30 thêm ~0,2 điểm cho cả ba, RTF ×5 → chọn beam 10 làm thiết lập chính là hợp lý.
- ⚠️ Số relPE in ở đây trong khi quy tắc trục GPU-giờ (TODO ⛔ 13/10) còn mở — relPE tốn GPU-giờ hơn hai bên
  Mamba; khi so phải nêu điều kiện.

## 3.2 CR-CTC — số liệu đã kiểm từ bài (arXiv 2410.05101v3, Zipformer, CTC → CR-CTC, cùng chi phí train)

Bài train CR-CTC với **nửa batch + nửa số epoch** so với CTC (hai bản augment/câu → cùng chi phí); α = 0,2; số vùng
time mask và tỉ lệ che tối đa ×2,5 (ablation: 4,98 → 4,62 test-other). Chỉ thử trên Zipformer — **không có số cho Mamba**.
- Aishell-1 (~170 h, gần cỡ dữ liệu ta) Zipformer-S 23,1M, 120 → 60 epoch: test 5,26 → 4,12 (−22% tương đối).
- GigaSpeech (podcast/YouTube) S 22,1M: test 11,95 → 11,58 (−3%). LibriSpeech S: test-other 6,89 → 5,85 (−15%).
- Ước cho ta (ngoại suy, chưa đo): −3% đến −22% tương đối WER greedy; có thể ~0 với Mamba.
- Chi phí nếu theo đúng thiết kế bài (19 epoch, batch 8 câu × 2 bản): ≈ run 40 ms hiện có (4,9–6,0 h/mô hình) → ~17–19
  giờ quota cho 3 mô hình + preflight.

Tiếng Anh trong clean-test (93 câu có `corrected_at`): 3,4% số từ không phải âm tiết Việt (bar, stress, friendzone,
bartender…); WER nhãn giả so với bản hiệu đính 2,1%.

## 4. Đã trao đổi với người dùng (cách báo cáo)

Beam + LM làm **output hệ thống** tốt hơn thật, nhưng **không làm encoder tốt hơn**; LM mang kiến thức văn bản ngoài
và có thể che khác biệt encoder; LM học nhãn giả Zipformer; từ điển chặn từ ngoài từ điển (tên riêng). Đề xuất: bảng
chính RQ1 giữ greedy; bảng phụ beam + từ điển + LM áp cho cả 5 mô hình; muốn mô hình tốt hơn thật thì phải sửa phía
train (BPE3000 / CR-CTC). Người dùng chưa chọn.
