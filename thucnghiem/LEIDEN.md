# Leiden backend

Thuật toán được đề xuất và tất cả các biến thể GA hoàn toàn viết bằng ngôn ngữ Julia. Đối với đường cơ sở Leiden, dự án cố ý sử dụng cầu nối PythonCall tùy chọn vì một API Leiden thuần túy, ổn định và hiện đại của Julia chưa nằm trong lõi của API Graphs.jl.

Cài đặt trong Julia:
```Julia
using Pkg
Pkg.add("PythonCall")
```

Sau đó, hãy đảm bảo môi trường Python mà PythonCall nhìn thấy có chứa python-igraph và leidenalg. Nếu Leiden chưa được cài đặt, hãy chạy thí nghiệm chính với tham số `--skip-leiden`; tuyệt đối không âm thầm thay thế bằng Louvain mà lại gán nhãn là Leiden.