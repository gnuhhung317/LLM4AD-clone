# Phân Tích Chi Tiết Luồng Thực Thi MEoH (Multi-objective Evolution of Heuristic)

## Tổng Quan Kiến Trúc MEoH

MEoH là một framework tiến hóa đa mục tiêu sử dụng Large Language Model để tự động thiết kế heuristic. Framework này áp dụng các nguyên lý evolutionary computing với LLM để tạo ra các heuristic chất lượng cao cho các bài toán optimization phức tạp.

### Thành Phần Chính

1. **Population**: Quần thể các heuristic function
2. **MEoHSampler**: Module sampling sử dụng LLM  
3. **MEoHPrompt**: Hệ thống prompt template cho các toán tử tiến hóa
4. **SecureEvaluator**: Module đánh giá an toàn cho các function
5. **MEoHProfiler**: Hệ thống logging và profiling

## Luồng Thực Thi Chi Tiết - Method `run()`

### Phase 1: Khởi Tạo (Initialization)

```python
def run(self):
    if not self._resume_mode:
        # do init
        self._population = Population(pop_size=self._pop_size)
        self._init_population()
        while len([f for f in self._population if not np.isinf(np.array(f.score)).any()]) < self._selection_num:
            self._population._generation -= 1
            self._init_population()
```

#### 1.1 Tạo Population Rỗng
```python
self._population = Population(pop_size=self._pop_size)
```
- **Mục đích**: Khởi tạo quần thể với kích thước định trước
- **Cấu trúc**: 
  - `_population`: List các Function đã được đánh giá
  - `_next_gen_pop`: List các Function thế hệ tiếp theo
  - `_elitist`: List các Function Pareto-optimal
  - `_generation`: Số thế hệ hiện tại (bắt đầu từ 0)

#### 1.2 Gọi `_init_population()`

```python
def _init_population(self):
    # threads for sampling
    sampler_threads = [
        Thread(
            target=self._thread_init_population,
        ) for _ in range(self._num_samplers)
    ]
    for t in sampler_threads:
        t.start()
    for t in sampler_threads:
        t.join()
```

**Luồng thực thi**:
1. Tạo `self._num_samplers` threads song song
2. Mỗi thread chạy `_thread_init_population()`
3. Chờ tất cả threads hoàn thành (`join()`)

#### 1.3 Chi Tiết `_thread_init_population()`

```python
def _thread_init_population(self):
    while self._population.generation == 0:
        if not self._continue_sample():
            break
        try:
            # get a new func using i1
            prompt = MEoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
            self._sample_evaluate_register(prompt)
        except Exception as e:
            if self._debug_mode:
                traceback.print_exc()
                exit()
            continue
```

**Luồng chi tiết**:

1. **Kiểm tra điều kiện**: `self._population.generation == 0`
   - Chỉ chạy khi đang ở thế hệ khởi tạo

2. **Kiểm tra giới hạn sampling**: `self._continue_sample()`
   ```python
   def _continue_sample(self):
       if self._max_generations is None and self._max_sample_nums is None:
           return True
       if self._max_generations is None and self._max_sample_nums is not None:
           if self._tot_sample_nums < self._max_sample_nums:
               return True
           else:
               return False
       # ... logic kiểm tra khác
   ```

3. **Tạo I1 Prompt**: `MEoHPrompt.get_prompt_i1()`
   ```python
   def get_prompt_i1(cls, task_prompt: str, template_function: Function):
       prompt_content = f'''{task_prompt}
   1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
   2. Next, implement the following Python function:
   {str(temp_func)}
   Do not give additional explanations.'''
       return prompt_content
   ```

4. **Sample-Evaluate-Register**: `self._sample_evaluate_register(prompt)`

#### 1.4 Chi Tiết `_sample_evaluate_register(prompt)`

```python
def _sample_evaluate_register(self, prompt):
    sample_start = time.time()
    thought, func = self._sampler.get_thought_and_function(prompt)
    sample_time = time.time() - sample_start
    if thought is None or func is None:
        return

    # convert to Program instance
    program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
    if program is None:
        return

    # evaluate
    score, eval_time = self._evaluation_executor.submit(
        self._evaluator.evaluate_program_record_time,
        program
    ).result()

    # score
    func.score = score
    func.evaluate_time = eval_time
    func.algorithm = thought
    func.sample_time = sample_time
    
    try:
        if self._profiler is not None:
            self._profiler.register_function(func)
            if isinstance(self._profiler, MEoHProfiler):
                self._profiler.register_population(self._population)
            self._tot_sample_nums += 1
    except Exception as e:
        traceback.print_exc()

    # register to the population
    self._population.register_function(func)
```

**Luồng chi tiết từng bước**:

**Step 1: LLM Sampling**
```python
sample_start = time.time()
thought, func = self._sampler.get_thought_and_function(prompt)
sample_time = time.time() - sample_start
```

- Gọi `MEoHSampler.get_thought_and_function()`:
  ```python
  def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
      response = self._sampler.draw_sample(prompt)  # Gọi LLM API
      thought = self.__class__.trim_thought_from_response(response)  # Extract algorithm description từ {}
      code = SampleTrimmer.trim_preface_of_function(response)  # Extract function code
      function = SampleTrimmer.sample_to_function(code, self._template_program)  # Convert to Function object
      if function is not None:
          function.entire_code = str(SampleTrimmer.sample_to_program(code, self._template_program))
      return thought, function
  ```

**Step 2: Code Conversion**
```python
program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
```
- Convert Function object thành Program object để có thể execute

**Step 3: Parallel Evaluation**
```python
score, eval_time = self._evaluation_executor.submit(
    self._evaluator.evaluate_program_record_time,
    program
).result()
```
- Submit evaluation task to ThreadPoolExecutor/ProcessPoolExecutor
- `SecureEvaluator.evaluate_program_record_time()` thực hiện:
  - Chạy program với test cases
  - Đo thời gian execution
  - Trả về score (single value hoặc multi-objective array)

**Step 4: Function Annotation**
```python
func.score = score
func.evaluate_time = eval_time
func.algorithm = thought
func.sample_time = sample_time
```

**Step 5: Profiling & Logging**
```python
if self._profiler is not None:
    self._profiler.register_function(func)
    if isinstance(self._profiler, MEoHProfiler):
        self._profiler.register_population(self._population)
    self._tot_sample_nums += 1
```

**Step 6: Population Registration**
```python
self._population.register_function(func)
```

#### 1.5 Chi Tiết `Population.register_function(func)`

```python
def register_function(self, func: Function):
    if func.score is None:
        return
    try:
        self._lock.acquire()  # Thread-safe
        # register to next_gen
        if not self.has_duplicate_function(func):
            self._next_gen_pop.append(func)

        # update: perform survival if reach the pop size
        if len(self._next_gen_pop) >= self._pop_size or (len(self._next_gen_pop) >= self._pop_size // 4 and self._generation == 0):
            pop = self._population + self._next_gen_pop

            # Elitist Update - Pareto Front Calculation
            pop_elitist = pop + self._elitist
            objs = [ind.score for ind in pop_elitist]
            objs_array = -np.array(objs)  # Convert to maximization
            nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)
            self._elitist = []
            for idx in nondom_idx.tolist():
                self._elitist.append(pop_elitist[idx])

            # Dominance-Dissimilarity Selection
            crt_pop_size = len(pop)
            dominated_counts = np.zeros((crt_pop_size, crt_pop_size))
            for i in range(crt_pop_size):
                for j in range(i + 1, crt_pop_size):
                    if (np.array(pop[i].score) >= np.array(pop[j].score)).all():
                        dominated_counts[i, j] = -calc_syntax_match([pop[i].entire_code], pop[j].entire_code, 'python')
                    elif (np.array(pop[j].score) >= np.array(pop[i].score)).all():
                        dominated_counts[j, i] = -calc_syntax_match([pop[j].entire_code], pop[i].entire_code, 'python')
            
            dominated_counts_ = dominated_counts.sum(0)
            self._population = [pop[i] for i in np.argsort(-dominated_counts_)[:self._pop_size // 5]]
            self._next_gen_pop = []
            self._generation += 1
```

**Cơ chế Dominance-Dissimilarity**:
1. **Dominance**: So sánh objective values
2. **Dissimilarity**: Sử dụng syntax matching để đo độ khác biệt code
3. **Selection**: Chọn individuals có dominance cao và dissimilarity cao

#### 1.6 Kiểm Tra Feasible Population

```python
while len([f for f in self._population if not np.isinf(np.array(f.score)).any()]) < self._selection_num:
    self._population._generation -= 1
    self._init_population()
```

- **Mục đích**: Đảm bảo có đủ feasible solutions để thực hiện selection
- **Logic**: Nếu số lượng function có score hợp lệ < `selection_num`, reset generation và init lại

### Phase 2: Evolution Process

```python
# do evolve
self._do_sample()
```

#### 2.1 `_do_sample()` Method

```python
def _do_sample(self):
    sampler_threads = [
        Thread(
            target=self._thread_do_evolutionary_operator,
        ) for _ in range(self._num_samplers)
    ]
    for t in sampler_threads:
        t.start()
    for t in sampler_threads:
        t.join()
```

- Tạo `self._num_samplers` threads chạy song song
- Mỗi thread thực hiện `_thread_do_evolutionary_operator()`

#### 2.2 `_thread_do_evolutionary_operator()` - Core Evolution Loop

```python
def _thread_do_evolutionary_operator(self):
    while self._continue_sample():
        try:
            # E1 Operator - Crossover với diversification
            indivs = [self._population.selection() for _ in range(self._selection_num)]
            prompt = MEoHPrompt.get_prompt_e1(self._task_description_str, indivs, self._function_to_evolve)
            self._sample_evaluate_register(prompt)
            if not self._continue_sample():
                break

            # E2 Operator - Crossover với guided evolution
            if self._use_e2_operator:
                indivs = [self._population.selection() for _ in range(self._selection_num)]
                prompt = MEoHPrompt.get_prompt_e2(self._task_description_str, indivs, self._function_to_evolve)
                self._sample_evaluate_register(prompt)
                if not self._continue_sample():
                    break

            # M1 Operator - Mutation with modification
            if self._use_m1_operator:
                indiv = self._population.selection()
                prompt = MEoHPrompt.get_prompt_m1(self._task_description_str, indiv, self._function_to_evolve)
                self._sample_evaluate_register(prompt)
                if not self._continue_sample():
                    break

            # M2 Operator - Parameter-based mutation
            if self._use_m2_operator:
                indiv = self._population.selection()
                prompt = MEoHPrompt.get_prompt_m2(self._task_description_str, indiv, self._function_to_evolve)
                self._sample_evaluate_register(prompt)
                if not self._continue_sample():
                    break
```

**Chi tiết các Evolution Operators**:

#### **E1 Operator - Diversification Crossover**

**Selection Phase**:
```python
indivs = [self._population.selection() for _ in range(self._selection_num)]
```

**Population.selection() Method**:
```python
def selection(self) -> Function:
    funcs = [f for f in self._population if not np.isinf(np.array(f.score)).any()]
    
    if len(funcs) > 0:
        crt_pop_size = len(funcs)
        dominated_counts = np.zeros((crt_pop_size, crt_pop_size))
        for i in range(crt_pop_size):
            for j in range(i + 1, crt_pop_size):
                if (np.array(funcs[i].score) >= np.array(funcs[j].score)).all():
                    dominated_counts[i, j] = -calc_syntax_match([funcs[i].entire_code], funcs[j].entire_code, 'python')
                elif (np.array(funcs[j].score) >= np.array(funcs[i].score)).all():
                    dominated_counts[j, i] = -calc_syntax_match([funcs[j].entire_code], funcs[i].entire_code, 'python')
        dominated_counts_ = dominated_counts.sum(0)
        p = np.exp(dominated_counts_) / np.exp(dominated_counts_).sum()
    
    return np.random.choice(funcs, p=p, replace=False)
```

- **Cơ chế**: Probabilistic selection dựa trên dominance-dissimilarity scores
- **Mục đích**: Chọn individuals có balance giữa quality và diversity

**Prompt Generation**:
```python
prompt = MEoHPrompt.get_prompt_e1(self._task_description_str, indivs, self._function_to_evolve)
```

```python
def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function):
    prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones. 
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
    return prompt_content
```

- **Mục đích**: Tạo algorithm hoàn toàn khác biệt với các individuals hiện có
- **Strategy**: Diversification để tránh premature convergence

#### **E2 Operator - Guided Evolution Crossover**

```python
def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function):
    prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
    return prompt_content
```

- **Mục đích**: Guided evolution dựa trên common patterns
- **Strategy**: Exploitation of good patterns while maintaining diversity

#### **M1 Operator - Modification Mutation**

```python
def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function):
    prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
    return prompt_content
```

- **Mục đích**: Local search around good solutions
- **Strategy**: Modification-based mutation

#### **M2 Operator - Parameter Mutation**

```python
def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function):
    prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
    return prompt_content
```

- **Mục đích**: Parameter-level fine-tuning
- **Strategy**: Parameter space exploration

### Phase 3: Termination & Cleanup

```python
# finish
if self._profiler is not None:
    self._profiler.finish()
```

#### 3.1 Executor Shutdown

Trong `_thread_do_evolutionary_operator()`:
```python
# shutdown evaluation_executor
try:
    self._evaluation_executor.shutdown(cancel_futures=True)
except:
    pass
```

#### 3.2 Profiler Finalization

```python
if self._profiler is not None:
    self._profiler.finish()
```

## Cơ Chế Đồng Bộ & Thread Safety

### 1. Population Thread Safety

```python
class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        # ...
        self._lock = Lock()

    def register_function(self, func: Function):
        try:
            self._lock.acquire()
            # ... population operations
        finally:
            self._lock.release()
```

### 2. Parallel Evaluation

```python
# Sử dụng ThreadPoolExecutor hoặc ProcessPoolExecutor
if multi_thread_or_process_eval == 'thread':
    self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=num_evaluators
    )
else:
    self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=num_evaluators
    )
```

### 3. Multi-threaded Sampling

- `num_samplers` threads chạy song song cho sampling
- Mỗi thread độc lập thực hiện evolution operators
- Synchronization qua shared Population object

## Điểm Mạnh Của Thiết Kế

### 1. Scalability

- **Parallel Sampling**: Multiple sampler threads
- **Parallel Evaluation**: ThreadPool/ProcessPool cho evaluation
- **Async Processing**: Non-blocking evaluation submission

### 2. Robustness

- **Exception Handling**: Comprehensive try-catch blocks
- **Graceful Degradation**: Continue on individual failures
- **Resource Management**: Proper executor shutdown

### 3. Flexibility

- **Configurable Operators**: Toggle E2, M1, M2 operators
- **Adaptive Population**: Dynamic size management
- **Multi-objective Support**: Built-in Pareto optimization

### 4. Quality Assurance

- **Dominance-Dissimilarity**: Quality + diversity balance
- **Duplicate Detection**: Prevent redundant individuals
- **Syntax Validation**: Ensure code correctness

## Phân Tích Performance

### 1. Bottlenecks Tiềm Năng

- **LLM API Calls**: Network latency và rate limits
- **Code Evaluation**: Complex test cases
- **Syntax Matching**: Expensive for large populations
- **Non-dominated Sorting**: O(MN²) complexity

### 2. Optimization Strategies

- **Caching**: Duplicate function detection
- **Batch Processing**: Multiple evaluations per submission
- **Lazy Evaluation**: Defer expensive operations
- **Population Size Tuning**: Balance quality vs speed

## Kết Luận

MEoH framework thể hiện một thiết kế sophisticated cho evolutionary algorithm với LLM integration. Luồng thực thi được tối ưu cho:

1. **Parallel Processing**: Maximizing throughput
2. **Quality Control**: Ensuring solution quality
3. **Diversity Maintenance**: Preventing premature convergence
4. **Robustness**: Handling failures gracefully

Framework này đặc biệt hiệu quả cho các bài toán optimization phức tạp where traditional heuristic design requires domain expertise sâu.