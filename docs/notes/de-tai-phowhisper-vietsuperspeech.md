# Ghi nhớ đề tài khóa luận: Fine-tune PhoWhisper cho tiếng Việt hội thoại tự nhiên

Cập nhật: 2026-08-14

## Bối cảnh lựa chọn

- Bậc: Cử nhân, thời gian: 15 tuần, phải chắc chắn hoàn thành.
- Ràng buộc compute: chỉ Google Colab free (T4) hoặc laptop cá nhân.
- Đã loại các hướng trước đó:
  - Mamba/SSM cho RAG tiếng Việt: bị loại vì checkpoint Mamba pretrained chỉ có tiếng Anh, dùng cho tiếng Việt tạo confound nghiên cứu (cần continued pretraining, rủi ro cao). Người dùng từ chối hướng cần continued pretrain.
  - Các hướng train-from-scratch so sánh kiến trúc (Mamba vs Transformer): bị bỏ khi người dùng quyết định mở rộng lại toàn bộ hướng AI thay vì chỉ NLP/LLM.
  - Đề tài dạng xây "Đánh giá"/evaluator (self-explanation faithfulness): bị từ chối vì người dùng không có nền tảng về methodology đánh giá.
- Sở thích đã xác nhận: "mới" = được đề xuất gần đây (recency-based), không cần risk-taking/liên ngành. Ưu tiên đề tài có ý tưởng hơn plain fine-tuning, có kết quả kiểm chứng/demo được.
- Có sẵn bộ dữ liệu VietSuperSpeech muốn tận dụng.

## Đề tài đã chốt để tiếp tục thảo luận

**Fine-tune PhoWhisper cho nhận dạng tiếng Việt hội thoại tự nhiên (casual/conversational), dùng bộ dữ liệu VietSuperSpeech, kèm phân tích lỗi có hệ thống.**

### Vấn đề nghiên cứu

ASR tiếng Việt hiện tại (kể cả PhoWhisper - SOTA) chủ yếu mạnh trên giọng đọc chuẩn/tin tức/audiobook. Với giọng nói đời thường (nói nhanh, chêm từ đệm, tiếng lóng, ngắt quãng, giọng vùng miền) WER tăng rõ rệt. VietSuperSpeech là dữ liệu hội thoại đời thường lấy từ 4 kênh YouTube, phù hợp để domain-adapt.

### Câu hỏi nghiên cứu (RQ)

- RQ1: Fine-tune PhoWhisper trên VietSuperSpeech cải thiện WER bao nhiêu so với PhoWhisper gốc, đo trên test set hội thoại tự nhiên?
- RQ2: Loại lỗi nào giảm nhiều nhất sau fine-tune, loại nào vẫn tồn tại dai dẳng (tên riêng, số liệu, từ vay mượn, chồng tiếng)?
- RQ3 (tùy chọn, thêm chiều sâu): LoRA fine-tuning có đạt hiệu năng tương đương full fine-tuning trong khi tốn ít tài nguyên hơn không?

### Dataset

- VietSuperSpeech: 52,023 cặp audio-text, 267.39 giờ, tiếng Việt hội thoại đời thường (chuyện phiếm, vlog, cộng đồng Việt kiều, bình luận phi chính thức). 16kHz mono WAV, câu dài 3-30s. Split sẵn: 46,822 train (240.67h) / 5,201 dev-test (26.72h), seed cố định. Nguồn: huggingface.co/datasets/thanhnew2001/VietSuperSpeech.
- Lưu ý minh bạch quan trọng: transcript là pseudo-label, sinh tự động bằng Zipformer-30M-RNNT-6000h qua Sherpa-ONNX, KHÔNG được người kiểm tra tay.
- Kế hoạch xử lý: dùng toàn bộ train set để fine-tune (chấp nhận nhiễu nhãn ở mức vừa phải - bình thường trong ASR). Tự nghe và sửa tay một tập con nhỏ từ dev-test (đề xuất 200-300 câu, lấy mẫu ngẫu nhiên có seed cố định) làm "clean test set" riêng để báo cáo WER cuối cùng đáng tin cậy hơn. Ghi rõ quy trình lấy mẫu, tiêu chí sửa trong báo cáo.

### Mô hình & phương pháp

- Base model: PhoWhisper (VinAI) - đã pretrained/finetuned sẵn cho tiếng Việt (844 giờ dữ liệu đa dạng), SOTA trên benchmark ASR tiếng Việt, ICLR 2024 Tiny Paper (arXiv 2406.02555). Giải quyết đúng vấn đề cross-lingual confound mà Mamba gặp phải vì đã "biết" tiếng Việt.
- Size đề xuất: PhoWhisper-small hoặc PhoWhisper-medium, dùng LoRA/PEFT để fine-tune - đủ nhẹ chạy nhiều epoch trên T4 free, đủ mạnh để kết quả có ý nghĩa.
- Tiền lệ kỹ thuật đã xác nhận khả thi: HuggingFace PEFT có notebook chính thức fine-tune whisper-large-v2 bằng LoRA + 8-bit quantization trên free T4, dưới 8GB VRAM. Repo tham khảo: fast-whisper-finetuning (Vaibhavs10), finetune-whisper-lora (fengredrum). Đã có người làm LoRA cho tiếng Việt: doof-ferb/whisper-large-peft-lora-vi (dùng làm related work/đối chiếu pipeline).
- Không cần compile CUDA kernel custom (khác Mamba) - rủi ro kỹ thuật thấp hơn nhiều.

### Phân tích lỗi (taxonomy) - phần tạo chiều sâu học thuật, thay thế cho "xây evaluator"

Phân loại lỗi ASR thành khoảng 5-6 nhóm, đếm tần suất trước/sau fine-tune trên cùng test set:
1. Từ đệm/chêm bị bỏ sót hoặc nhận sai (ừ, kiểu, gì đó...)
2. Từ lóng/khẩu ngữ bị đoán thành từ chuẩn gần giống
3. Tên riêng/địa danh
4. Số đếm
5. Lỗi do chồng tiếng hoặc nói quá nhanh
6. Lỗi ngắt câu/dấu câu

Đây là gán nhãn quan sát thủ công trên mẫu vừa đủ, không đòi hỏi kiến thức đánh giá phức tạp.

### Demo cho buổi phản biện

Giao diện Gradio: upload/thu âm đoạn hội thoại tiếng Việt đời thường → chạy song song PhoWhisper gốc vs bản fine-tune → hiển thị 2 transcript cạnh nhau, highlight khác biệt, show WER nếu người dùng nhập transcript đúng.

### Rủi ro

- Rủi ro chính: thời gian train nhiều epoch trên ~47K mẫu trên Colab free - cần ước lượng giờ cần và có phương án checkpoint/resume nếu session bị ngắt.
- Không phải rủi ro về tính đúng đắn nghiên cứu (khác trường hợp Mamba).

### Cấu trúc chương báo cáo (dự kiến)

1. Giới thiệu & vấn đề
2. Related Work (Whisper, PhoWhisper, các dataset ASR tiếng Việt khác)
3. Phương pháp (mô hình, LoRA, dữ liệu, taxonomy lỗi)
4. Thực nghiệm & Kết quả (WER, phân tích lỗi theo nhóm, so sánh LoRA vs full nếu làm RQ3)
5. Demo
6. Kết luận & hướng phát triển

### Tiềm năng paper

Hướng "domain adaptation cho ASR hội thoại tự nhiên tiếng Việt kèm phân tích lỗi có hệ thống" có thể viết thành short paper cho workshop tiếng Việt/NLP khu vực (vd. VLSP workshop).

### Tài liệu tham khảo đã xác nhận

- PhoWhisper: Automatic Speech Recognition for Vietnamese - https://arxiv.org/abs/2406.02555
- vinai/PhoWhisper-large (Hugging Face) - https://huggingface.co/vinai/PhoWhisper-large
- VietSuperSpeech dataset - https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech
- Fine-Tune Whisper with Transformers and PEFT - https://github.com/fengredrum/finetune-whisper-lora
- fast-whisper-finetuning - https://github.com/Vaibhavs10/fast-whisper-finetuning
- doof-ferb/whisper-large-peft-lora-vi - https://huggingface.co/doof-ferb/whisper-large-peft-lora-vi
- LoRA - Hugging Face PEFT docs - https://huggingface.co/docs/peft/main/en/conceptual_guides/lora

## Trạng thái

Đề tài đã được thảo luận kỹ, đã có bản đề cương sơ bộ chính thức (docx). Đang chờ gửi GVHD.

## Cập nhật 2026-08-23 — đã cân nhắc và giữ nguyên đề tài

Đã yêu cầu phân tích phản biện (khả thi/tính mới/đóng góp/có quá đơn giản như đồ án môn học) và tìm đề tài thay thế. Kết luận phản biện:

- Khả thi: chấp nhận được, điểm nghẽn thật là tuần 5-6 (fine-tune 240h audio trên Colab free chỉ 2 tuần) — coi đây là vùng đệm chính, không phải tuần 11/15.
- Tính mới: yếu, dựa chủ yếu vào "chưa ai kết hợp PhoWhisper + VietSuperSpeech" (recency-based, đã xác nhận chấp nhận được).
- Ranh giới với đồ án môn học: phần taxonomy lỗi + clean test set tự làm là thứ cứu đề tài khỏi mức "chạy lại tutorial fine-tune" — nhưng chỉ nếu chương 4 có thảo luận/lý giải sâu (không chỉ đếm bảng số liệu).

Đã tìm 3 hướng thay thế (ASR theo phương ngữ vùng miền dùng dataset ViMD/EMNLP 2024 hoặc Vietnamese Regional Voice Dataset 2025; nén mô hình PhoWhisper/PhoBERT qua quantization cho thiết bị biên; RAG pháp lý/y tế tiếng Việt — bị loại vì đụng lại vấn đề thiếu nền tảng methodology đánh giá, tương tự lý do loại hướng evaluator trước đó).

**Quyết định: giữ nguyên đề tài PhoWhisper/VietSuperSpeech**, không đổi hướng. Việc cần làm để bù lại tính mới yếu: không để RQ3 (LoRA vs full fine-tune) là optional-rồi-bỏ nếu hết giờ — nếu đến tuần 11 không kịp làm RQ3, bắt buộc bù bằng cách đào sâu phân tích lỗi hơn (ví dụ tương quan giữa độ dài câu/tốc độ nói và tỷ lệ lỗi) thay vì chỉ báo cáo tần suất lỗi theo nhóm, để tránh chương kết quả bị mỏng.
