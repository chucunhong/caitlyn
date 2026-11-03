# CAITLYN: code thực nghiệm

Code đi kèm `paper/main.tex` và `paper/supplementary.tex`. Thiết kế theo `IMPLEMENTATION_GUIDE.md`. LLM gọi qua **Alibaba Cloud Model Studio (DashScope)**, dùng endpoint tương thích OpenAI.

## 1. Cài đặt

```powershell
# Khuyến nghị đặt venv ở đường dẫn chỉ có ký tự ASCII, ví dụ C:\venvs\caitlyn
python -m venv C:\venvs\caitlyn
C:\venvs\caitlyn\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # rồi điền DASHSCOPE_API_KEY
```

- **LightGBM báo `Could not find module lib_lightgbm.dll`**: máy thiếu *Microsoft Visual C++ Redistributable 2015–2022 (x64)*, cụ thể là `vcomp140.dll`. Cài từ trang của Microsoft là hết lỗi.
- Model mặc định là `qwen-plus` (`configs/run.yaml` → `llm.model`). Khi chạy chính thức cho bài, nên ghim snapshot có ngày (ví dụ `qwen-plus-2025-xx-xx`) và sửa macro `\backbone` trong `main.tex` cho khớp.
- Tài khoản ở Trung Quốc đại lục: đặt `DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` trong `.env`.
- Mọi lời gọi LLM được cache trong `results/llm_cache.sqlite`. Chạy lại hoặc chạy ablation dùng chung stage sẽ không tốn thêm tiền. Không xóa file này giữa các lần chạy.
- Bất kỳ tham số nào trong `run.yaml` đều ghi đè được bằng `--set`, ví dụ `--set caitlyn.B=50 llm.max_concurrency=4`.

## 2. Kiểm tra nhanh không tốn tiền (mock LLM)

```powershell
python scripts/prepare_data.py --dataset all
python scripts/run_caitlyn.py --dataset german --seeds 0 --mock --m 200 --set caitlyn.B=10 caitlyn.M=20 caitlyn.J=2
```

Kết quả mock nằm trong các thư mục `*__mock`. `make_tables.py` và `make_figures.py` luôn bỏ qua chúng, trừ khi đặt `CAITLYN_INCLUDE_MOCK=1`. **Xóa các thư mục `*__mock` trước khi chạy thật.**

## 3. Thứ tự chạy thật (mục 11 của guide)

| Bước | Lệnh | Cho ra |
|---|---|---|
| 0 | `python scripts/prepare_data.py --dataset all` | `data/processed/*` (Credit scoring: xem mục 4) |
| 1 | `python scripts/estimate_cost.py` | ước tính lời gọi, token, US$ trước khi chạy |
| 3 | `python scripts/run_caitlyn.py --dataset adult` (rồi `credit_scoring`, `german`) | dòng CAITLYN trong Bảng 2–3 |
| 4 | `python scripts/run_baseline.py --dataset <ds> --method <m>` với `real reweigh ctgan decaf great tabfairgan structsynth cllm` | các dòng baseline |
| 5 | Ablation (Bảng 6): `--variant nogate / pcplan / noproxy / perattr / rllm / rempty / fullhist / lam0` trên `adult` và `credit_scoring` | `tab_ablation_rows` |
| 6 | Seesaw (Bảng 5): `--variant none`; `--variant perattr --order sex age race` và `--order age sex race` (Credit scoring chỉ có 2 thuộc tính: `--order sex age` và `--order age sex`) | `tab_seesaw_rows` |
| 7 | Leakage (Hình 2b), trên `adult` và `credit_scoring`: `run_caitlyn.py --dataset <ds>` với `--variant leak_full`, `leak_parent`, `leak_indep` (bản `full` đã có ở bước 3), rồi `python scripts/run_leakage.py --dataset <ds>` | `fig_analysis_ab` (b) |
| 8 | Sensitivity trên Adult: `--n-train 100/500/2000`, `--lam 0.0/0.6/1.0`, `--k 1/2`, `--delta`, `--tau-p`, `--model open` | `fig_sensitivity` |
| 9 | `python scripts/evaluate.py [--bootstrap]` | `metrics.json` và `preds.csv` cho mọi lần chạy |
| 10 | `python scripts/make_tables.py` rồi `python scripts/make_figures.py` | `paper/generated/*.tex`, `paper/figures/*.pdf` |

`main.tex` và `supplementary.tex` nạp các file sinh ra qua `\InputIfExists` / `\IfFileExists`. Khi chưa có kết quả, bài vẫn biên dịch được với bảng `\TBD`. Khi file đã được sinh, bảng thật tự thay vào chỗ. `paper/generated/numbers.tex` chứa macro cho các con số chính (`\resSPCaitlynC`, `\resAUCAdult`, …) để dùng khi viết các đoạn `\todo{results}`. Các đoạn văn mô tả kết quả và câu kết quả trong abstract vẫn phải tự viết sau khi có số.

**Baseline chạy từ repo riêng** (TabSyn, FairTabDDPM, SENNA, Golob et al., CLLM bản chính thức):
```powershell
python scripts/run_baseline.py --dataset adult --method export     # results/adult/_export/<seed>/
# ... chạy repo ngoài trên train.csv ...
python scripts/run_baseline.py --dataset adult --method import --name tabsyn --seed 0 --file path\to\synth.csv
```
DECAF cần plan của CAITLYN cùng seed, nên chạy CAITLYN trước. TabFairGAN cần clone repo chính thức vào `baselines/external/TabFairGAN`. StructSynth là bản cài lại: thay prompt trong `caitlyn/prompts/structsynth_gen.txt` bằng prompt trong phụ lục của bài StructSynth.

## 4. Việc còn lại phải tự làm

- **Dữ liệu** lấy từ `../zeri/data` (đã chép vào `data/raw/`). Credit scoring bảo vệ **sex + age** (ngưỡng = trung vị 36 tuổi). `Marital` không có codebook công khai nên được giữ là cột thường với mã gốc, giống SENNA, lucian và zeri. Sex: 1 = nam, 2 = nữ (Giang et al. 2024).
- Kiểm tra giá token trong `run.yaml` (`price_in_per_m`, `price_out_per_m`) trước khi báo cáo bảng chi phí.
- **Cấu hình mặc định (đã ghi vào bài):** 3 seed; |D_synth| = 5.000 dòng cho mọi phương pháp và mọi dataset; lô 50 dòng mỗi lời gọi; Stage D gom lời gọi theo tầng topo (mỗi tầng hai nhóm, A và ngoài A). Biến thể `--variant percol` (mỗi cột một lời gọi, như bản cũ) có thể chạy thêm làm ablation.
- **Không có dữ liệu hay SCM nào do mình tự định nghĩa trước.** Graph được LLM dựng từ đầu trên dữ liệu thật (Stage A, kiểu StructSynth). Mọi đánh giá đều chạy trên ba dataset thật; không còn CreditSCM, Bảng 4 (mechanism) hay SHD trên bnlearn.
- **Ước tính chi phí** (`python scripts/estimate_cost.py`): khoảng 190k lời gọi, 490M token vào và 90M token ra, **≈ $350** cho toàn bộ kế hoạch thí nghiệm ở giá qwen-plus $0,40/$1,20 (cận trên, chưa trừ phần cache).

## 5. Chỗ code khác với guide (đã sửa trong bài)

1. **Số hoán vị B.** Với B = 200, p-value nhỏ nhất là 1/201. Sau hiệu chỉnh Holm trên m kiểm định (39 trên Adult), không cạnh nào qua được evidence gate. Code dùng `B = max(200, ⌈2m/α⌉)` cho mỗi họ kiểm định; `main.tex` và Table A đã được sửa theo.
2. **BFS:** khi hàng đợi rỗng, các cột chưa thăm được lần lượt đưa vào, để π_role không phải liệt kê cột ngoại sinh. Nhãn Y là nút lá (`label_is_sink`).
3. **German:** cột `personal_status` được thay bằng `sex` (nếu giữ lại thì nó là proxy hoàn hảo của sex). Mã A-code được dịch sang chữ để LLM hiểu.
4. **NN privacy risk:** holdout là tập test (30%), nên train được lấy mẫu con cho bằng kích thước holdout.

## 6. Cấu trúc

```
caitlyn/            llm.py (DashScope, cache, retry), data.py, assoc.py, common.py,
                    stage_a_plan.py, stage_b_proxy.py, stage_c_surgery.py, stage_d_generate.py,
                    pipeline.py (biến thể + cache stage), prompts/*.txt, eval/*
baselines/          reweighing, dgm (CTGAN/DECAF/GReaT), tabfairgan, llm_baselines (StructSynth, CLLM), external
scripts/            prepare_data, estimate_cost, run_caitlyn, run_baseline, evaluate, run_leakage,
                    make_tables, make_figures
results/<ds>/<method>/<seed>/   synth.csv, run.json, plan_edited.json, metrics.json, preds.csv
results/<ds>/_stages/           artifact của Stage A–C (dùng chung giữa các biến thể)
```
