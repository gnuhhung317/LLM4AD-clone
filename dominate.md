# Dominance-Dissimilarity Selection trong MEoH Framework

## Tổng Quan

File này giải thích chi tiết cơ chế **Dominance-Dissimilarity Selection** trong hàm `register_function()` của MEoH framework - một thuật toán quan trọng để duy trì diversity và chất lượng trong population của các heuristic functions.

## Phân Tích Chi Tiết Hàm `register_function()`

### 1. **Thread-Safe Registration**

```python
def register_function(self, func: Function):
    if func.score is None:
        return
    try:
        self._lock.acquire()  # Thread-safe
```

**Mục đích**: Đảm bảo thread safety khi multiple samplers cùng lúc thêm functions vào population.

**Tại sao cần thiết**: MEoH sử dụng multi-threading với nhiều `sampler_threads` chạy song song, nên cần synchronization để tránh race conditions.

### 2. **Duplicate Detection & Registration**

```python
# register to next_gen
if not self.has_duplicate_function(func):
    self._next_gen_pop.append(func)
```

**Chức năng**: 
- Kiểm tra function đã tồn tại trong population chưa
- Chỉ thêm functions mới vào `_next_gen_pop` (generation tiếp theo)

**Lợi ích**: Tránh redundancy, maintain diversity trong population

### 3. **Survival Trigger Conditions**

```python
if len(self._next_gen_pop) >= self._pop_size or (len(self._next_gen_pop) >= self._pop_size // 4 and self._generation == 0):
```

**Điều kiện kích hoạt survival selection**:
- **Điều kiện chính**: `len(_next_gen_pop) >= _pop_size` → Population đầy
- **Điều kiện đặc biệt**: Generation 0 chỉ cần 25% population size

**Tại sao generation 0 khác**:
- Generation đầu cần khởi tạo population nhanh
- Tránh chờ quá lâu để bắt đầu evolution process

## Elitist Update - Pareto Front Calculation

### 4. **Combine Populations**

```python
pop = self._population + self._next_gen_pop

# Elitist Update - Pareto Front Calculation  
pop_elitist = pop + self._elitist
```

**Quy trình**:
1. **`pop`**: Kết hợp current population + new generation
2. **`pop_elitist`**: Thêm cả elitist individuals từ các generations trước

### 5. **Non-Dominated Sorting**

```python
objs = [ind.score for ind in pop_elitist]
objs_array = -np.array(objs)  # Convert to maximization
nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)
```

**Chi tiết**:
- **Extract objectives**: Lấy scores từ tất cả individuals
- **Convert to maximization**: `-np.array(objs)` vì PyMOO expects minimization
- **Non-dominated sorting**: Tìm Pareto front (solutions không bị dominate)

**Ví dụ VRPTW 2-objective**:
```python
# Individual A: [-100, -50]  (distance=-100, violations=-50)
# Individual B: [-120, -40]  (distance=-120, violations=-40)  
# Individual C: [-90, -60]   (distance=-90, violations=-60)

# After convert: [[100, 50], [120, 40], [90, 60]]
# Non-dominated: A và B (C bị A dominate)
```

### 6. **Update Elitist Archive**

```python
self._elitist = []
for idx in nondom_idx.tolist():
    self._elitist.append(pop_elitist[idx])
```

**Chức năng**: Lưu trữ các solutions tốt nhất (non-dominated) từ tất cả generations để:
- Tránh mất thông tin quý giá
- Maintain long-term memory của evolution process

## Dominance-Dissimilarity Selection

### 7. **Dominance Matrix Calculation**

```python
crt_pop_size = len(pop)
dominated_counts = np.zeros((crt_pop_size, crt_pop_size))
for i in range(crt_pop_size):
    for j in range(i + 1, crt_pop_size):
        if (np.array(pop[i].score) >= np.array(pop[j].score)).all():
            dominated_counts[i, j] = -calc_syntax_match([pop[i].entire_code], pop[j].entire_code, 'python')
        elif (np.array(pop[j].score) >= np.array(pop[i].score)).all():
            dominated_counts[j, i] = -calc_syntax_match([pop[j].entire_code], pop[i].entire_code, 'python')
```

**Algorithm Logic**:

1. **Dominance Check**: `(np.array(pop[i].score) >= np.array(pop[j].score)).all()`
   - Individual i dominates j nếu i tốt hơn hoặc bằng j trên TẤT CẢ objectives
   - Và tốt hơn j trên ÍT NHẤT 1 objective

2. **Syntax Similarity Penalty**: `-calc_syntax_match(...)`
   - Tính độ tương đồng AST giữa 2 code functions
   - **Negative sign**: Càng giống nhau càng bị penalty
   - **Mục đích**: Maintain diversity trong population

**Ví dụ Cụ Thể**:
```python
# Function A dominates Function B với similarity = 0.8
dominated_counts[A, B] = -0.8

# Function C dominates Function D với similarity = 0.3  
dominated_counts[C, D] = -0.3

# Function A có lợi thế hơn C (ít similar = better diversity)
```

### 8. **Selection Based on Dominated Counts**

```python
dominated_counts_ = dominated_counts.sum(0)
self._population = [pop[i] for i in np.argsort(-dominated_counts_)[:self._pop_size // 5]]
```

**Selection Strategy**:

1. **Sum columns**: `dominated_counts.sum(0)`
   - Mỗi individual có tổng điểm từ việc bị dominated bởi others
   - Điểm càng âm = bị dominated nhiều + high similarity

2. **Argsort descending**: `np.argsort(-dominated_counts_)`
   - Sắp xếp theo thứ tự giảm dần của dominated_counts
   - Chọn individuals ít bị dominated nhất + most diverse

3. **Select top performers**: `[:self._pop_size // 5]`
   - Chỉ giữ lại 20% population size tốt nhất
   - **Tại sao 20%**: Cân bằng between exploitation và exploration

### 9. **Cleanup & Generation Update**

```python
self._next_gen_pop = []
self._generation += 1
```

**Finalization**:
- Clear next generation buffer
- Increment generation counter
- Chuẩn bị cho evolution cycle tiếp theo

## Ví Dụ Minh Họa

### Scenario: VRPTW với 4 functions

```python
# Population scores (distance, violations):
Function_A: [-100, -50]  # Good distance, average violations
Function_B: [-120, -40]  # Poor distance, good violations  
Function_C: [-90, -60]   # Average distance, poor violations
Function_D: [-95, -45]   # Balanced performance

# Syntax similarities:
sim(A,B) = 0.8  # Very similar code
sim(A,C) = 0.3  # Different approaches
sim(A,D) = 0.5  # Moderately similar
sim(B,C) = 0.2  # Very different
sim(B,D) = 0.4  # Moderately similar  
sim(C,D) = 0.6  # Quite similar
```

**Dominance Analysis**:
```python
# A dominates C: [-100,-50] >= [-90,-60] → A tốt hơn both objectives
dominated_counts[A,C] = -0.3

# B dominates C: [-120,-40] >= [-90,-60] → B có violations tốt hơn, distance tương đương
# Không dominate vì B kém hơn distance

# D không dominate ai, không bị dominate
```

**Final Selection**:
- **Function A**: Good performance + reasonable diversity → Selected
- **Function D**: Balanced performance + good diversity → Selected  
- **Function B**: Bị loại vì performance kém
- **Function C**: Bị loại vì bị dominated

## Lợi Ích của Thuật Toán

### 1. **Multi-objective Optimization**
- Pareto front maintenance đảm bảo không mất solutions tốt
- Non-dominated sorting tìm trade-offs optimal

### 2. **Diversity Preservation**  
- AST similarity penalty tránh population convergence sớm
- Maintain multiple promising approaches

### 3. **Elitist Strategy**
- Lưu trữ best solutions across generations
- Tránh genetic drift và loss of good solutions

### 4. **Computational Efficiency**
- Chỉ select 20% population → Reduce computational cost
- Thread-safe operations cho parallel execution

### 5. **Adaptive Selection Pressure**
- Generation 0: Lower threshold (25%) cho quick initialization
- Later generations: Full population requirement cho thorough selection

## Kết Luận

Thuật toán **Dominance-Dissimilarity Selection** trong MEoH framework là một innovation quan trọng, kết hợp:

- **Pareto optimality** cho multi-objective optimization
- **Code diversity** thông qua AST similarity analysis  
- **Elitist preservation** cho long-term memory
- **Efficient selection** cho computational scalability

Đây là core mechanism giúp MEoH tạo ra các heuristic functions vừa high-quality vừa diverse, essential cho success của automated algorithm design trong LLM4AD platform.
