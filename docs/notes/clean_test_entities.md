# Bảng tên riêng chuẩn cho clean-test (2026-10-14)

Dùng khi hiệu đính (`clean_test_correction_guide.md` mục 2a). Ứng viên lấy từ output Whisper large-v3
(`data/processed/whisper_advisor.json`), nhãn giả và bản hiệu đính, gom theo video. **Cách viết lấy theo nguồn
ngoài, không theo Whisper**: Whisper sai ở vài tên (cột cuối).

Ghi vào nhãn bằng **chữ thường** (WER chuẩn hoá về chữ thường). Ký hiệu như `&` bị `normalize_text` đổi thành
khoảng trắng: `p&g` và `p g` chấm như nhau. Cách viết thay thế (`pg`, `p and g`) xử lý lúc chấm, không phải lúc ghi.

## 1. Đã xác minh từ nguồn

| Video (câu) | Ghi là | Nguồn | Whisper / nhãn giả viết |
|---|---|---|---|
| Ruby Nguyễn (#9, #12) | curieous | [Vietcetera VIV S2 EP25](https://vietcetera.com/vn/video/vietnam-innovators-tieng-viet/bao-gio-cho-het-chuyen-doi-moi-ruby-nguyen-ceo-curieous-viv-s2-ep25) | **Curious** (sai) / cua, crush |
| OPES (#15) | opes | [Vietcetera S3#26](https://vietcetera.com/vn/podcast/vietnam-innovators-tieng-viet/s326-dang-hoang-tung-ceo-opes-bao-hiem-so-thuan-tien-va-de-tiep-can) | **OPEX** (sai) / ope, opec |
| OPES (#34) | miro (host Miro Nguyễn) | như trên | Miro |
| P&G (#58, #59) | p&g | tiêu đề video "…tại P&G…, P&G Vietnam" | P&G, PNG / bnj, p |
| Gojek (#13, #39) | gojek | tiêu đề video "Giám đốc Marketing Gojek Vietnam" | Gojek / câu trách, gojec |
| Thảo Đan (#85) | suboi | [Vietcetera MAD EP20](https://vietcetera.com/vn/video/mad/ke-chuyen-bang-hinh-anh-3d-thao-dan-visual-artist-va-creative-director-mad-ep20) (MV "No Nê" của Suboi) | Suboi / suboy |
| Maika Elan (#161) | maika, the pink choice | [World Press Photo 2013](https://www.worldpressphoto.org/collection/photo-contest/2013/maika-elan/9) | **Micah** (sai), The Pink Choice / mai ca, the pin choi |
| Cởi Mở (#158) | cởi mở đi unitour | [Vietcetera, Cởi Mở Đi Unitour](https://vietcetera.com/vn/bo-suu-tap/coi-mo-di-unitour) | Unitour / unitur |
| Dustin (#179, #190) | bar stories | [Vietcetera](https://vietcetera.com/loanh-quanh-an-choi-nghi-ve-doi-cung-dustin-phuc-nguyen/) (talk show trên kênh Dustin on the Go) | Bar Stories / bastoris, bastori |
| Lên Men (#107) | vietcetera | tên kênh | **Vietcera** (sai) |

## 2. Tên phổ biến, cách viết chuẩn quen thuộc (chưa tra riêng cho video)

axie infinity (#11), kardiachain (tiêu đề video), smartosc (tiêu đề video), kristin neff, self compassion (#74),
evolutionary psychology (#72), silicon valley, wall street, hollywood, new york (#28), new york university, u penn (người dùng chốt 15/10, ghi hai chữ),
harvard (#10; đã sửa *upan* → *u penn* 15/10), indonesia (#32), singapore (#35), forbes, florida (#88),
aig, chubb (#17, hãng bảo hiểm; Zipformer ghi *ag*), murakami, kafka (#176), hikikomori (#166), trịnh công sơn
(#150), head and shoulders (#196, nhãn hàng), dna, nft, mv.

## 3. Cần nghe lại, chưa xác minh

- #161 *chú nick wood* (Whisper): có thể là **Nick Út**, nhiếp ảnh gia chiến tranh, vì câu nói về "khoảnh khắc
  chiến tranh". Đây là phỏng đoán, nghe lại rồi mới ghi.
- #159 *Glory Hall* / *clori ho*. #196 *UBA*: Whisper ghi *Unaided Brand Awareness*.
- Các chỗ Zipformer phiên âm sai mà Whisper đưa được phương án: #114 *tipping point*, #35 *bachelor*,
  #56 *inspiration*, #152 *interdependence*, #117 *onboarding*, #118 *customer success / account manager*.
  Phải nghe để quyết.

## 4. Câu Whisper bịa, bỏ qua cố vấn

Ở #60, #89, #125, #145, #147, #195, Whisper large-v3 ra *"Hãy subscribe cho kênh Ghiền Mì Gõ để không bỏ lỡ những
video hấp dẫn"*, có câu lặp ba lần. Đây là ảo giác của Whisper, học từ câu chào cuối video YouTube, không liên
quan đến audio. Công cụ đánh dấu các câu này và không tô đối chiếu.
