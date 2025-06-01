# Nghiên cứu Task VRPTW_Construct và Framework MEoH

## Tổng quan

Vehicle Routing Problem with Time Windows (VRPTW) là một bài toán tối ưu hóa phức tạp trong lĩnh vực logistics và vận chương. Task `vrptw_construct` trong LLM4AD framework được thiết kế để sử dụng Large Language Models (LLMs) để tự động thiết kế các thuật toán heuristic xây dựng cho bài toán VRPTW.

## 1. Mô tả bài toán VRPTW

### 1.1 Định nghĩa bài toán
VRPTW là một biến thể của Vehicle Routing Problem (VRP) với ràng buộc thời gian:

**Đầu vào:**
- Một depot (kho chính)
- Tập hợp các khách hàng với tọa độ, nhu cầu, cửa sổ thời gian và thời gian phục vụ
- Đội xe có cùng dung lượng

**Ràng buộc:**
- Xe xuất phát và trở về depot
- Mỗi khách hàng được phục vụ đúng một lần
- Thỏa mãn tất cả nhu cầu
- Không vượt quá dung lượng xe
- Tuân thủ ràng buộc cửa sổ thời gian

**Mục tiêu:** Tối thiểu hóa tổng khoảng cách di chuyển và vi phạm thời gian

### 1.2 Phiên bản đa mục tiêu (VRPTW_Construct_2O)
Framework hỗ trợ phiên bản đa mục tiêu với hai mục tiêu:
1. **Mục tiêu 1:** Tổng khoảng cách di chuyển (cần tối thiểu hóa)
2. **Mục tiêu 2:** Tổng vi phạm/chờ đợi thời gian (cần tối thiểu hóa)

## 2. Kiến trúc hệ thống

### 2.1 Cấu trúc thư mục
```
llm4ad/task/optimization/vrptw_construct_2O/
├── __init__.py          # Export VRPTWEvaluation2O
├── evaluation.py        # Lớp đánh giá chính
├── get_instance.py      # Tạo dữ liệu bài toán
├── template.py          # Template hàm và mô tả task
└── paras.yaml          # Tham số cấu hình
```

### 2.2 Các thành phần chính

#### 2.2.1 GetData Class (`get_instance.py`)
```python
class GetData:
    def __init__(self, n_instance, n_cities):
        self.n_instance = n_instance    # Số lượng instance
        self.n_cities = n_cities        # Số khách hàng
        self.max_time = 4.6            # Thời gian tối đa
```

**Chức năng:** Tạo dữ liệu ngẫu nhiên cho bài toán VRPTW

**Quy trình tạo instance:**
1. **Tạo tọa độ:** `coordinates = np.random.rand(self.n_cities + 1, 2)`
2. **Tạo nhu cầu:** `demands = np.append(np.array([0]), np.random.randint(1, 10, size=self.n_cities))`
3. **Tính ma trận khoảng cách:** `distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)`
4. **Tạo thời gian phục vụ:** `node_serviceTime = np.random.rand(self.n_cities) * 0.05 + 0.15`
5. **Tạo cửa sổ thời gian:**
   ```python
   node_lengthTW = np.random.rand(self.n_cities) * 0.05 + 0.15
   d0i = distances[0][1:]
   ei = np.random.rand(self.n_cities) * (((4.6 * np.ones(self.n_cities) - node_serviceTime - node_lengthTW) / d0i - 1) - 1) + 1
   node_earlyTW = np.multiply(ei, d0i)
   node_lateTW = node_earlyTW + node_lengthTW
   ```

#### 2.2.2 VRPTWEvaluation2O Class (`evaluation.py`)
Lớp đánh giá chính kế thừa từ `Evaluation` base class.

**Constructor:**
```python
def __init__(self, timeout_seconds=30, problem_size=50, n_instance=16, **kwargs):
```

**Phương thức chính:**

1. **`tour_cost_and_violations()`:** Tính toán chi phí và vi phạm của một lộ trình
```python
def tour_cost_and_violations(self, distance_matrix, solution, time_service, time_windows):
    total_distance = 0
    total_violations = 0
    current_time = 0
    
    for j in range(len(solution) - 1):
        travel_time = distance_matrix[int(solution[j]), int(solution[j + 1])]
        current_time += travel_time
        
        arrival_time = current_time
        earliest_time = time_windows[solution[j + 1]][0]
        latest_time = time_windows[solution[j + 1]][1]
        
        # Tính thời gian chờ (đến sớm)
        waiting_time = max(0, earliest_time - arrival_time)
        current_time = max(arrival_time, earliest_time)
        
        # Tính vi phạm (đến muộn)
        violation = max(0, arrival_time - latest_time)
        
        total_distance += travel_time
        total_violations += waiting_time + violation
        
        current_time += time_service[solution[j + 1]]
        
        if solution[j + 1] == 0:  # Quay về depot
            current_time = 0
    
    return total_distance, total_violations
```

2. **`evaluate()`:** Đánh giá một heuristic function
   - Chạy thuật toán trên tất cả instances
   - Kiểm tra tính khả thi của solution
   - Trả về điểm số đa mục tiêu `np.array([-avg_distance, -avg_violations])`

#### 2.2.3 Template Function (`template.py`)
```python
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, 
                    rest_capacity: np.ndarray, current_time: np.ndarray,
                    demands: np.ndarray, distance_matrix: np.ndarray, 
                    time_windows: np.ndarray) -> int:
```

**Tham số đầu vào:**
- `current_node`: ID của node hiện tại
- `depot`: ID của depot
- `unvisited_nodes`: Array các node chưa thăm
- `rest_capacity`: Dung lượng còn lại
- `current_time`: Thời gian hiện tại
- `demands`: Nhu cầu của các node
- `distance_matrix`: Ma trận khoảng cách
- `time_windows`: Cửa sổ thời gian

## 3. Framework MEoH (Multi-objective Evolutionary optimization of Heuristics)

### 3.1 Kiến trúc MEoH
MEoH là một framework tối ưu hóa đa mục tiêu sử dụng thuật toán tiến hóa để phát triển các heuristic function.

#### 3.1.1 MEoH Class (`meoh.py`)
**Constructor chính:**
```python
def __init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase = None,
             max_generations: int = 10, max_sample_nums: int = 100, 
             pop_size: int = 20, selection_num=5, num_objs: int = 2, ...):
```

**Các tham số quan trọng:**
- `max_generations`: Số thế hệ tối đa
- `max_sample_nums`: Số mẫu tối đa
- `pop_size`: Kích thước quần thể
- `selection_num`: Số cá thể được chọn cho crossover
- `num_objs`: Số mục tiêu (=2 cho VRPTW)

#### 3.1.2 Population Class (`population.py`)
Quản lý quần thể các heuristic functions:

**Non-dominated Sorting:**
```python
objs_array = -np.array(objs)
nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)
```

**Selection với AST Similarity:**
```python
def selection(self) -> Function:
    funcs = [f for f in self._population if not np.isinf(np.array(f.score)).any()]
    
    dominated_counts = np.zeros((crt_pop_size, crt_pop_size))
    for i in range(crt_pop_size):
        for j in range(i + 1, crt_pop_size):
            if (np.array(funcs[i].score) >= np.array(funcs[j].score)).all():
                dominated_counts[i, j] = -calc_syntax_match([funcs[i].entire_code], funcs[j].entire_code, 'python')
    
    p = np.exp(dominated_counts_) / np.exp(dominated_counts_).sum()
    return np.random.choice(funcs, p=p, replace=False)
```

#### 3.1.3 MEoHPrompt Class (`prompt.py`)
Tạo các prompt cho LLM với 5 loại operator:

1. **I1 (Initialization):** Khởi tạo algorithm mới
2. **E1 (Evolution 1):** Tạo algorithm khác biệt hoàn toàn
3. **E2 (Evolution 2):** Tạo algorithm dựa trên backbone idea
4. **M1 (Mutation 1):** Biến đổi algorithm hiện có
5. **M2 (Mutation 2):** Thay đổi tham số algorithm

### 3.2 MEoHSampler Class (`sampler.py`)
Xử lý sampling từ LLM:
```python
def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
    response = self._sampler.draw_sample(prompt)
    thought = self.trim_thought_from_response(response)
    code = SampleTrimmer.trim_preface_of_function(response)
    function = SampleTrimmer.sample_to_function(code, self._template_program)
    return thought, function
```

## 4. Luồng hoạt động tổng thể

### 4.1 Quy trình chính của MEoH
```
1. Khởi tạo
   ├── Tạo template program từ evaluation
   ├── Khởi tạo population rỗng
   ├── Tạo sampler và evaluator
   └── Cấu hình multi-threading

2. Khởi tạo Population (I1 operator)
   ├── Tạo prompt khởi tạo
   ├── LLM generate functions
   ├── Evaluate functions
   └── Thêm vào population

3. Evolution Loop
   ├── E1: Tạo algorithm hoàn toàn mới
   ├── E2: Tạo algorithm từ backbone idea  
   ├── M1: Biến đổi algorithm hiện có
   ├── M2: Thay đổi tham số algorithm
   └── Lặp lại cho đến khi đạt điều kiện dừng

4. Selection và Survival
   ├── Non-dominated sorting
   ├── AST similarity calculation
   └── Cập nhật population
```

### 4.2 Chi tiết evaluation process
```python
def evaluate(self, heuristic):
    distances = np.ones(self.n_instance)
    violations = np.ones(self.n_instance)
    
    for instance, distance_matrix, demands, vehicle_capacity, time_service, time_windows in self._datasets:
        route = []
        current_load = 0
        current_node = 0
        current_time = 0
        route.append(current_node)
        unvisited_nodes = set(range(1, self.problem_size + 1))
        
        while unvisited_nodes:
            # Gọi heuristic function được LLM generate
            next_node = heuristic(current_node, 0, feasible_unvisited_nodes,
                                vehicle_capacity - current_load, current_time,
                                copy.deepcopy(demands), copy.deepcopy(distance_matrix),
                                copy.deepcopy(time_windows))
            
            if next_node == 0:  # Quay về depot
                route.append(next_node)
                current_load = 0
                current_time = 0
                current_node = 0
            else:  # Đi đến khách hàng
                travel_time = distance_matrix[current_node, next_node]
                current_time += travel_time
                
                # Kiểm tra time window
                if current_time < time_windows[next_node][0]:
                    current_time = time_windows[next_node][0]
                if max(current_time, time_windows[next_node][0]) > time_windows[next_node][1]:
                    return None  # Infeasible solution
                
                current_time += time_service[next_node]
                current_load += demands[next_node]
                current_node = next_node
                route.append(current_node)
                unvisited_nodes.remove(next_node)
            
            # Cập nhật feasible nodes
            feasible_unvisited_nodes = []
            for node in unvisited_nodes:
                travel_time_to_node = distance_matrix[current_node, node]
                arrival_time = current_time + travel_time_to_node
                
                if (demands[node] <= vehicle_capacity - current_load and
                    arrival_time <= time_windows[node][1]):
                    feasible_unvisited_nodes.append(node)
            
            if not feasible_unvisited_nodes and unvisited_nodes:
                feasible_unvisited_nodes = [0]  # Buộc quay về depot
        
        # Tính điểm cho instance này
        total_distance, total_violations = self.tour_cost_and_violations(
            distance_matrix, route, time_service, time_windows)
        
        distances[n_ins] = total_distance
        violations[n_ins] = total_violations
    
    # Trả về điểm đa mục tiêu (âm vì MEoH maximize)
    avg_distance = np.average(distances)
    avg_violations = np.average(violations)
    return np.array([-avg_distance, -avg_violations])
```

### 4.3 Multi-threading Architecture
MEoH sử dụng multi-threading để tăng hiệu suất:

```python
# Evaluation threads
self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_evaluators)

# Sampling threads  
sampler_threads = [Thread(target=self._thread_do_evolutionary_operator) 
                  for _ in range(self._num_samplers)]
```

## 5. Ví dụ sử dụng

### 5.1 Cấu hình và chạy MEoH
```python
from llm4ad.task.optimization.vrptw_construct_2O import VRPTWEvaluation2O
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.meoh import MEoH, MEoHProfiler

# Cấu hình LLM
llm = HttpsApi(host='api.openai.com', key='sk-xxx', model='gpt-4', timeout=60)

# Cấu hình task
task = VRPTWEvaluation2O(timeout_seconds=30, problem_size=50, n_instance=16)

# Cấu hình MEoH
method = MEoH(
    llm=llm,
    evaluation=task,
    profiler=MEoHProfiler(log_dir='logs/meoh_vrptw'),
    max_generations=10,
    max_sample_nums=100,
    pop_size=20,
    num_evaluators=4,
    num_objs=2
)

# Chạy evolution
method.run()
```

### 5.2 Ví dụ heuristic function được generate
```python
def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, current_time,
                    demands, distance_matrix, time_windows):
    """
    Hybrid time-distance heuristic với capacity consideration
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    best_node = None
    best_score = float('inf')
    
    for node in unvisited_nodes:
        if demands[node] > rest_capacity:
            continue
            
        travel_time = distance_matrix[current_node][node]
        arrival_time = current_time + travel_time
        
        # Time window penalty
        early_penalty = max(0, time_windows[node][0] - arrival_time)
        late_penalty = max(0, arrival_time - time_windows[node][1]) * 10
        
        # Distance and urgency factors
        urgency = 1.0 / (time_windows[node][1] - current_time + 0.1)
        
        score = travel_time + early_penalty + late_penalty - urgency * 0.5
        
        if score < best_score:
            best_score = score
            best_node = node
    
    return best_node if best_node is not None else depot
```

## 6. Kết luận

Framework MEoH kết hợp với task VRPTW_Construct tạo ra một hệ thống mạnh mẽ để tự động thiết kế các thuật toán heuristic cho bài toán VRPTW. Các điểm mạnh chính:

1. **Tự động hóa:** Sử dụng LLM để generate algorithms thay vì thiết kế thủ công
2. **Đa mục tiêu:** Tối ưu đồng thời distance và time violations
3. **Đa dạng hóa:** Các operator khác nhau tạo ra algorithms đa dạng
4. **Hiệu suất:** Multi-threading cho evaluation và sampling
5. **Tính mở rộng:** Dễ dàng thêm constraints hoặc objectives mới

Framework này mở ra hướng nghiên cứu mới trong việc ứng dụng AI để thiết kế thuật toán tối ưu hóa, đặc biệt hữu ích cho các bài toán phức tạp như VRPTW trong thực tế.