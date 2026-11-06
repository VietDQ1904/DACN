# MO-HGA-LS — Julia experimental implementation

Đây là bản viết lại phần thực nghiệm theo đúng kiến trúc đang mô tả trong `dacn(2).tex`, với core MO-HGA-LS viết thuần Julia để giảm overhead của vòng lặp GA/Local Search.

## Thiết kế giữ nguyên

- LAR + CC-biased initialization, `rho=0.7`.
- Uniform crossover `pc=0.85`.
- Neighbor mutation `pm=1/n`.
- Tournament selection `t=2`.
- GA / HGA / NSGA-II / MO-HGA-LS.
- HGA = single-objective Q + Local Search.
- NSGA-II = two objectives Q/Phi, no Local Search.
- MO-HGA-LS = two objectives Q/Phi + Pareto-Lamarck Local Search.
- Reencode bằng BFS.
- Pareto archive `A_max=100`.
- Hypervolume trong `(1-Q, Phi)`, reference `(1.1,1.1)`.
- `G_max=200`, `k_stag=30`, `eps_stop=1e-4`, `I_LS=30`, `p_LS=0.2`.
- NMI/ARI chỉ đánh giá, không tham gia tối ưu.

## Dataset chính theo báo cáo

`karate polbooks football email amazon`.

Amazon đầy đủ chỉ chạy Louvain/Leiden vì GA không phù hợp với quy mô 334k đỉnh. Polblogs không nằm trong main experiment; nếu cần có thể thêm vào `--datasets` như một phụ lục riêng.

Email được báo cáo theo **đồ thị sau preprocessing** (undirected, bỏ self-loop, LCC), đồng thời metadata raw nên được ghi riêng trong báo cáo.

## Yêu cầu

CSV = "0.10"
CodecZlib = "0.7"
DataFrames = "1"
Graphs = "1.15"
JSON3 = "1"
julia = "1.10"

## Cài đặt

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Leiden là backend tùy chọn. Core GA/MO-HGA-LS không phụ thuộc Python. Nếu cần Leiden đúng baseline, cài `PythonCall` và trong Python environment có `igraph` + `leidenalg`.

## Kiểm thử

```bash
julia --project=. scripts/check.jl
```

## Chạy thực nghiệm chính

Nên chạy single-thread để benchmark runtime công bằng; dùng nhiều Julia threads khi chỉ cần throughput.

```bash
JULIA_NUM_THREADS=1 julia --project=. scripts/run_experiments.jl \
  --datasets karate,polbooks,football,email \
  --algos GA,HGA,NSGA2,MO-HGA-LS,Louvain,Leiden \
  --runs 30 --out results/main
```

Nếu chạy multi-thread để tính toán nhanh hơn và dùng GPU thông qua CUDA:

```bash
JULIA_NUM_THREADS=20 julia --project=. scripts/run_experiments.jl   --datasets karate,polbooks,football,email   --algos GA,HGA,NSGA2,MO-HGA-LS,Louvain,Leiden   --runs 30 --out results/main --cuda
```

Nếu chạy theo phạm vi seed (ví dụ chạy seed từ 10 đến 20):

```bash
JULIA_NUM_THREADS=20 julia --project=. scripts/run_experiments.jl   --datasets karate,polbooks,football,email   --algos GA,HGA,NSGA2,MO-HGA-LS,Louvain,Leiden --runs 10 --seed-start 10 --seed-end 20 --out results/main
```

Nếu chạy với tham số đầu vào nhẹ hơn (phù hợp hơn với dữ liệu lớn như Amazon):

```bash
JULIA_NUM_THREADS=20 julia --project=. scripts/run_experiments.jl   --datasets amazon --algos GA,HGA,NSGA2,MO-HGA-LS,Louvain,Leiden   --runs 30 --out results/main --preset-fast
```


## Phân tích:

```bash
julia --project=. scripts/analyze.jl results/main/runs.jsonl results/amazon/runs.jsonl --out results/report
```

## Profile thời gian và bộ nhớ sử dụng:

```bash
julia --project=. scripts/profile.jl --dataset email --algo MO-HGA-LS --seed 1 --repeats 1
```

## Local Search ablation

```bash
for pls in 0.0 0.1 0.2 0.5 1.0; do
  for ils in 1 5 10 30; do
    julia --project=. scripts/run_experiments.jl \
      --datasets football --algos MO-HGA-LS --runs 10 \
      --p_ls $pls --i_ls $ils --out results/sweep_pls_${pls}_ils_${ils}
done; done
```

Không gọi kết quả sweep là "tối ưu" nếu chưa định nghĩa protocol chọn cấu hình. Nếu dùng sweep để hiệu chỉnh, chọn cấu hình trên tập calibration rồi **đóng băng** trước khi chạy 5 dataset đánh giá.

## Output

- `runs.jsonl`: một dòng cho mỗi `(dataset, algorithm, seed)`.
- `config.json`: cấu hình bất biến của lần chạy.
- `results/report/tables/*.csv`: bảng trung bình, độ lệch chuẩn, median, Q1/Q3.
- `dataset_metadata_postprocessed.csv`: thống kê đúng graph đưa vào thuật toán.

## Lưu ý reproducibility

Không gộp các run có `mu/lam/g_max/p_ls/i_ls` khác nhau vào cùng một bảng. Một bảng chính phải có cùng configuration key. HV chỉ dùng để so sánh các thuật toán đa mục tiêu (`NSGA2` và `MO-HGA-LS`); GA/HGA nên so sánh Q/NMI/ARI/runtime.
