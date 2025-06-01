# Cách SecureEvaluator Đánh Giá Task - Hướng Dẫn Chi Tiết

## Tổng Quan

`SecureEvaluator` là lớp trung gian quan trọng trong LLM4AD framework, có nhiệm vụ đánh giá an toàn và hiệu quả các heuristic/algorithm được tạo ra bởi LLM. Nó đóng vai trò như một lớp bảo vệ (wrapper) xung quanh các lớp `Evaluation` cụ thể của từng task.

## Kiến Trúc Tổng Thể

```
[LLM Generated Code] → [SecureEvaluator] → [Task-Specific Evaluator] → [Score/Fitness]
```

### 1. **Input**: Code từ LLM
- Code string dạng function được LLM tạo ra
- Ví dụ: heuristic cho TSP, VRPTW, etc.

### 2. **Processing**: SecureEvaluator xử lý
- **Code modification**: Thêm các decorator, seed, protective div
- **Safe execution**: Chạy trong process riêng biệt
- **Timeout control**: Giới hạn thời gian thực thi

### 3. **Output**: Score từ Task Evaluator
- Fitness value cho single-objective
- Array of scores cho multi-objective (như VRPTW 3-objective)

## Quy Trình Đánh Giá Chi Tiết

### Bước 1: Code Preprocessing

```python
def _modify_program_code(self, program_str: str) -> str:
```

**Các biến đổi code:**

1. **Numba Acceleration** (nếu `use_numba_accelerate=True`)
   ```python
   # Before
   def my_heuristic(a, b):
       return a + b
   
   # After  
   import numba
   @numba.jit(nopython=True)
   def my_heuristic(a, b):
       return a + b
   ```

2. **Protected Division** (nếu `use_protected_div=True`)
   ```python
   # Before
   result = a / b
   
   # After
   result = _protected_div(a, b, delta=1e-5)
   
   def _protected_div(a, b, delta=1e-5):
       return a / (b + delta)  # Tránh chia cho 0
   ```

3. **Random Seed** (nếu `random_seed` được set)
   ```python
   # Before
   def my_heuristic():
       x = np.random.random()
       return x
   
   # After
   def my_heuristic():
       np.random.seed(2024)  # Thêm vào đầu function
       x = np.random.random()
       return x
   ```

### Bước 2: Safe Execution

```python
def evaluate_program(self, program: str | Program, **kwargs):
```

**Hai chế độ thực thi:**

#### A. Safe Mode (`safe_evaluate=True` - mặc định)

1. **Tạo Process riêng biệt**
   ```python
   process = multiprocessing.Process(
       target=self._evaluate_in_safe_process,
       args=(program_str, function_name, result_queue),
       daemon=self._evaluator.daemon_eval_process
   )
   ```

2. **Timeout Control**
   ```python
   if self._evaluator.timeout_seconds is not None:
       result = result_queue.get(timeout=self._evaluator.timeout_seconds)
   ```

3. **Process Management**
   - Terminate process sau khi hoàn thành
   - Kill process nếu timeout
   - Cleanup resources

#### B. Direct Mode (`safe_evaluate=False`)
- Chạy trực tiếp trong main process
- Không có timeout protection
- Rủi ro cao hơn nhưng performance tốt hơn

### Bước 3: Code Execution

```python
def _evaluate_in_safe_process(self, program_str: str, function_name, result_queue, **kwargs):
```

**Quá trình thực thi:**

1. **Compile Code**
   ```python
   all_globals_namespace = {}
   exec(program_str, all_globals_namespace)
   program_callable = all_globals_namespace[function_name]
   ```

2. **Call Task Evaluator**
   ```python
   res = self._evaluator.evaluate_program(program_str, program_callable, **kwargs)
   ```

3. **Return Result**
   ```python
   result_queue.put(res)
   ```

## Task-Specific Evaluation

Mỗi task có lớp `Evaluation` riêng kế thừa từ base class:

### Ví Dụ: VRPTW 3-Objective

```python
class VRPTWEvaluation3O(Evaluation):
    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)
    
    def evaluate(self, heuristic):
        # Chạy heuristic trên nhiều instances
        for instance_data in self._datasets:
            result = self.solve_single_instance(heuristic, instance_data)
            # Thu thập 3 objectives:
            # 1. Total distance
            # 2. Time violations  
            # 3. Execution time
        
        # Return as numpy array
        return np.array([-avg_distance, -avg_violations, -avg_execution_time])
```

**Quy trình đánh giá VRPTW:**

1. **Load Datasets**: Nhiều problem instances
2. **For each instance**:
   - Gọi heuristic để construct route
   - Đo thời gian thực thi algorithm
   - Tính total distance
   - Tính time window violations
3. **Aggregate**: Trung bình trên tất cả instances
4. **Return**: Array 3 objectives (âm vì MEoH minimize)

## Cấu Hình Evaluation

### Trong Constructor của Task

```python
class VRPTWEvaluation3O(Evaluation):
    def __init__(self, timeout_seconds=30, problem_size=50, n_instance=16, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,  # Có dùng Numba không
            timeout_seconds=timeout_seconds  # Timeout cho mỗi evaluation
        )
```

### Trong SecureEvaluator

```python
class SecureEvaluator:
    def __init__(self, evaluator: Evaluation, debug_mode=False, **kwargs):
        self._evaluator = evaluator
        self._debug_mode = debug_mode
        
        # Cấu hình multiprocessing method
        if self._evaluator.safe_evaluate:
            # MacOS/Linux dùng 'fork', Windows dùng 'spawn'
```

## Error Handling & Recovery

### 1. **Timeout Errors**
```python
except:
    # timeout
    if self._debug_mode:
        print(f'DEBUG: the evaluation time exceeds {self._evaluator.timeout_seconds}s.')
    process.terminate()
    result = None
```

### 2. **Compilation Errors**
```python
try:
    exec(program_str, all_globals_namespace)
except Exception as e:
    if self._debug_mode:
        print(e)
    result_queue.put(None)
```

### 3. **Runtime Errors**
```python
try:
    res = self._evaluator.evaluate_program(program_str, program_callable, **kwargs)
except Exception as e:
    return None  # Invalid solution
```

## Debug Mode

Khi `debug_mode=True`:

```python
if self._debug_mode:
    print(f'DEBUG: evaluated program:\n{program_str}\n')
    print(f'DEBUG: the evaluation time exceeds {self._evaluator.timeout_seconds}s.')
    print(e)  # Exception details
```

**Lợi ích:**
- Xem code sau khi modify
- Theo dõi timeout events  
- Debug compilation/runtime errors

## Performance Optimization

### 1. **Multiprocessing Configuration**
```python
# Linux/MacOS: fork (faster)
multiprocessing.set_start_method('fork', force=True)

# Windows: spawn (safer but slower)  
multiprocessing.set_start_method('spawn', force=True)
```

### 2. **Numba Acceleration**
- Tự động thêm `@numba.jit(nopython=True)` 
- Tăng tốc đáng kể cho numerical computation
- Trade-off: compilation overhead lần đầu

### 3. **Process Reuse**
- SecureEvaluator tạo process mới cho mỗi evaluation
- Đảm bảo isolation nhưng có overhead
- Future: có thể implement process pool

## Kết Luận

`SecureEvaluator` cung cấp:

- **Safety**: Isolation, timeout, error handling
- **Flexibility**: Hỗ trợ nhiều loại task khác nhau  
- **Performance**: Numba acceleration, multiprocessing
- **Debugging**: Chi tiết logs và error reporting

Đây là component core đảm bảo LLM-generated code được evaluate một cách an toàn và đáng tin cậy trong framework LLM4AD.
