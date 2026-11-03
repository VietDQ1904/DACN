# MO-HGA-LS – mã nguồn cho ĐACN "Phát hiện cộng đồng bằng Metaheuristic"

Cài đặt theo Chương 1–2 của báo cáo và đề cương Chương 3. Chỉ cần `numpy scipy networkx scikit-learn pandas matplotlib`;
Leiden cần thêm `igraph leidenalg`.

```bash
pip install -r requirements.txt
python tests/test_core.py && python tests/test_datasets.py    # kiểm thử (≈ 15 s)
python scripts/check_datasets.py                # kiểm tra tệp dữ liệu trong data/ (tải tay hoặc scripts/download_datasets.py)
```

## Cấu trúc

| Tệp | Nội dung | Mục trong báo cáo |
|---|---|---|
| `cdga/graph.py` | tiền xử lý (LCC, bỏ self-loop), CSR, CC, giải mã LAR, Q và Φ | 2.1, 2.2.1–2.2.3 |
| `cdga/lar.py` | khởi tạo CC-biased, uniform crossover, đột biến láng giềng, **Reencode** (BFS) | 2.2.2, 2.2.5, 2.2.6, 2.3.6, Alg. 3 |
| `cdga/local_search.py` | **Pareto-Lamarck LS**: Node Movement trên đỉnh ranh giới, ΔQ/ΔΦ gia tăng, archive cục bộ A_L | 2.3, Alg. 2 |
| `cdga/pareto.py` | trội Pareto, NSGA-II sort, crowding, Hypervolume, truncate, knee | 2.2.4, 2.2.7, 2.2.8 |
| `cdga/ga.py` | vòng lặp chính, archive, điều kiện dừng HV; 4 biến thể `GA`, `HGA`, `NSGA2`, `MO-HGA-LS` | 2.4, Alg. 1 |
| `cdga/baselines.py`, `metrics.py` | Louvain, Leiden, NMI, ARI | 2.5.3, 2.5.4 |
| `cdga/datasets.py`, `profile.py` | nạp Karate, Polbooks, Football, Polblogs, Email, Amazon, LFR; thống kê dữ liệu | 2.5.1 |
| `cdga/report_plots.py`, `report_plots2.py` | toàn bộ biểu đồ phân tích | 3.2–3.4 |
| `scripts/run_experiments.py` | chạy nhiều seed/cấu hình, ghi `runs.jsonl` (chạy tiếp được) | 3.2 |
| `scripts/check_datasets.py` | kiểm tra tệp dữ liệu và thống kê sau tiền xử lý | 2.5.1 |
| `scripts/analyze.py` | sinh bảng (`tables.md`, CSV) và hình, đánh số theo mục Chương 3 | 3.2–3.4 |

Bốn biến thể thuật toán (cùng LAR, cùng tham số chính):

| `--algos` | Mục tiêu | Local Search | Dùng ở mục |
|---|---|---|---|
| `GA` | Q | không | 3.2.1, 3.4.1 |
| `HGA` | Q | có | 3.2.2, 3.4.1 (MA-Q đơn mục tiêu) |
| `NSGA2` | Q và Φ | không (= MO-HGA-LS với p_LS = 0) | 3.2.1, 3.4.1 |
| `MO-HGA-LS` | Q và Φ | có | 3.2.2 |

## Dữ liệu (thư mục `data/`)

| Bộ dữ liệu | Tệp cần có | Nguồn |
|---|---|---|
| `karate` | không cần (có sẵn trong networkx; `karate.gml` của Newman không có nhãn nên không dùng) | – |
| `polbooks`, `football`, `polblogs` | `polbooks.gml`, `football.gml`, `polblogs.gml` | [Newman](https://public.websites.umich.edu/~mejn/netdata) (lấy tệp `.gml` trong zip) |
| `email` | `email-Eu-core.txt`, `email-Eu-core-department-labels.txt` (.gz cũng được) | [SNAP](https://snap.stanford.edu/data/email-Eu-core.html) |
| `amazon` | `com-amazon.ungraph.txt`, `com-amazon.all.dedup.cmty.txt` (hoặc `...top5000.cmty.txt`) | [SNAP](https://snap.stanford.edu/data/com-Amazon.html) |
| `amazon:top=K` | như `amazon` | đồ thị con từ K cộng đồng chuẩn lớn nhất, đủ nhỏ để chạy GA |

```bash
python scripts/check_datasets.py                          # tệp nào thiếu, nạp được chưa, n/m sau tiền xử lý
python scripts/check_datasets.py --datasets amazon:top=20
```
Amazon có nhãn chồng lấp: chỉ đỉnh thuộc đúng một cộng đồng có nhãn (còn lại -1), NMI/ARI chỉ tính trên các đỉnh có nhãn.
Đồ thị nhiều hơn `--max_ga_nodes` (mặc định 20000) tự bỏ qua các thuật toán GA; Amazon đầy đủ chỉ chạy Louvain/Leiden.

## Chạy thực nghiệm

```bash
# 3.2–3.3: so sánh chính (30 seed). Muốn đo thời gian chính xác dùng --jobs 1.
python scripts/run_experiments.py --datasets karate polbooks football polblogs email \
   --algos GA HGA NSGA2 MO-HGA-LS Louvain Leiden --runs 30 --jobs 4 --out results/main
# Email/Polblogs (n ≈ 1000) chậm hơn: dùng --preset fast (µ=λ=50, G_max=100, I_LS=5, p_LS=0.1)
python scripts/run_experiments.py --datasets email polblogs --preset fast --runs 20 --jobs 4 --out results/large
# Amazon: đầy đủ chỉ có Louvain/Leiden; bản con cho cả GA
python scripts/run_experiments.py --datasets amazon --algos Louvain Leiden --runs 5 --out results/amazon
python scripts/run_experiments.py --datasets amazon:top=20 --preset fast --runs 10 --out results/amazon_sub
# 3.4.2–3.4.3: khảo sát tham số Local Search
python scripts/run_experiments.py --datasets football --algos MO-HGA-LS --runs 10 \
   --grid "p_ls=0,0.1,0.2,0.5,1;i_ls=1,5,10,30" --jobs 4 --out results/sweep_ls
# bảng + hình (có thể gộp nhiều thư mục kết quả)
python scripts/analyze.py --runs results/main/runs.jsonl results/large/runs.jsonl results/amazon/runs.jsonl \
   --sweep results/sweep_ls/runs.jsonl --out results/report
```
Mọi tham số của Config đều đặt được bằng cờ (`--mu 100 --lam 100 --p_ls 0.2 --rho 0.7 --a_max 100 ...`).
LFR sinh tại chỗ: `--datasets "lfr:n=250,mu=0.3,seed=1"`.

## Kết quả phân tích sinh ra

`results/report/tables.md` (và `tables/*.csv`, mở được bằng Excel): Bảng 2.3 cập nhật số liệu thật, bảng 3.2.1–3.4.3, và một bảng phụ lục đầy đủ chỉ số cho từng bộ dữ liệu.

`results/report/figures/<bộ dữ liệu>/` (khoảng 17 hình mỗi bộ):

| Hình | Nội dung |
|---|---|
| `dataset_overview` | phân bố bậc, hệ số phân cụm, kích thước cộng đồng chuẩn, bảng thống kê (n, m, ⟨k⟩, CC, μ̂, Q và Φ chuẩn) |
| `dashboard`, `bars_metrics` | Q, Φ, NMI, ARI, K, thời gian, Hypervolume, số lần đánh giá theo thuật toán |
| `pareto`, `representative_solutions` | Pareto front Q–Φ; ba nghiệm đại diện maxQ / knee / minΦ |
| `convergence`, `anytime` | hội tụ theo thế hệ; Q và HV theo **thời gian chạy**; thời gian mỗi thế hệ |
| `time_cost`, `quality_vs_time` | cơ cấu thời gian (GA / Local Search), số lần đánh giá, chất lượng so với chi phí |
| `ls_ablation`, `ls_effect` | thêm Local Search thay đổi gì; tỉ lệ LS có tác dụng, ΔQ, ΔΦ, chi phí mỗi lần gọi |
| `wilcoxon`, `relative_scores`, `q_vs_nmi` | kiểm định cặp, điểm tương đối, Q cao có khớp nhãn chuẩn không |
| `network_partitions` | đồ thị tô màu theo nhãn chuẩn / Louvain / Leiden / MO-HGA-LS (n ≤ 1500) |
| `contingency`, `community_sizes` | ma trận so với nhãn chuẩn; phân bố kích thước cộng đồng |

`figures/summary/`: so sánh mọi bộ dữ liệu, hạng trung bình, thời gian theo quy mô đồ thị.
Hình dựa trên nhãn phân hoạch chỉ có khi n ≤ `--save_labels_max_n` (mặc định 3000); Amazon đầy đủ vì vậy chỉ có các hình số liệu.

## Quyết định cài đặt cần ghi/sửa lại trong báo cáo

1. **Số cộng đồng K cố định trong Local Search.** Phép chuyển bị cấm nếu |C_a| < 3 (để Reencode còn ≥ 2 đỉnh/cộng đồng) hoặc C_a∖{u} mất liên thông ⇒ không có cộng đồng rỗng, nên ΔΦ = (Δφ_a + Δφ_b)/K luôn đúng (khỏi cần "xử lý riêng" ở §2.3.4).
2. **Dừng sớm của LS khác dòng 15 của Thuật toán 2.** Dừng khi một vòng lặp ngoài *không làm A_L thay đổi* (thay vì cờ `moved`), vì chấp nhận các bước đi giữa nghiệm không bị trội có thể quay vòng mà không sinh nghiệm mới.
3. **Nếu LS không tìm được nghiệm khác phân hoạch ban đầu thì giữ nguyên kiểu gen gốc**, không Reencode (tránh làm mất đa dạng gen vô ích).
4. **K = 1** bị phạt (Q = −0.5, Φ = 1) để luôn bị trội; `SelectRepresentative` và knee dùng khoảng cách tới điểm lý tưởng sau chuẩn hóa min–max.
5. **N_high(i)** = láng giềng có CC ≥ CC trung bình của N(i). **Loại trùng phân hoạch** khi hợp R = P ∪ O; thiếu thì bù cá thể ngẫu nhiên CC-biased cho đủ µ.
6. **Hypervolume** trong không gian (1−Q, Φ), điểm tham chiếu cố định (1.1, 1.1). Archive: `A ← Truncate(ND(A ∪ O), A_max)` đúng Alg. 1.
7. Mặc định chưa hiệu chỉnh (báo cáo yêu cầu công bố rõ): λ = 100, p_LS = 0.2, ρ = 0.7, A_max = 100, k_stag = 30, ε_stop = 1e-4, p_m = 1/n.
8. NMI chuẩn hóa theo trung bình cộng (sklearn). Với Email, NMI/ARI dùng nhãn phòng ban (phân hoạch rời).

## Điều cần kiểm tra trên máy của bạn

* **Bảng 2.3, Email-Eu-core:** 25.571 là số cạnh *có hướng* trên SNAP (và 50,89 = 2×25571/1005). Sau khi vô hướng hóa, bỏ self-loop và lấy thành phần liên thông lớn nhất, số đỉnh/cạnh sẽ khác; `from_networkx` lưu thống kê thật trong `g.stats` để cập nhật bảng.
* **Amazon (334.863 đỉnh):** chi phí ~ G·(µ+λ)(n+m) ≈ 10¹⁰–10¹¹ bước, không khả thi cho GA bằng Python; chỉ chạy Louvain/Leiden hoặc bỏ khỏi so sánh GA.
* **Leiden:** cần `pip install igraph leidenalg` (môi trường tôi dùng không cài được nên nhánh `leidenalg` chưa được chạy thử). Không dùng `networkx.leiden_communities` vì trả về toàn đỉnh đơn lẻ.
* **Tốc độ:** Python thuần. LFR n = 250 (m = 1545), cấu hình mặc định, 1 lần MO-HGA-LS ≈ 140 s (114 thế hệ). Email (n ≈ 1000, m ≈ 16000) sẽ chậm hơn nhiều: nên hạ `--lam`, `--p_ls`, `--i_ls` hoặc chuyển vòng lặp LS sang numba/C++.
* Bộ nạp đã được kiểm thử bằng tệp MÔ PHỎNG đúng định dạng Newman/SNAP (tests/fixtures.py), chưa thử với tệp thật vì môi trường của tôi bị chặn tải. Chạy `scripts/check_datasets.py` trên máy bạn trước tiên.
