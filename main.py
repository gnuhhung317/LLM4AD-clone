from llm4ad.task.optimization.vrptw_construct_2O import VRPTWEvaluation2O
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.meoh import MEoH, MEoHProfiler

if __name__ == '__main__':
    llm = HttpsApi(
    )
    task = VRPTWEvaluation2O()
    # Cấu hình MEoH
    method = MEoH(
        llm=llm,
        evaluation=task,
        profiler=MEoHProfiler(
            log_dir='logs/meoh_vrptw-1',
            log_style='complex',
            evaluation_name='VRPTW',
            method_name='MEoH'
        ),
        max_generations=10,      # Số thế hệ tối đa
        max_sample_nums=200,     # Số sample tối đa
        pop_size=20,             # Kích thước population
        selection_num=5,         # Số cá thể được chọn trong crossover
        use_e2_operator=True,    # Sử dụng toán tử evolution e2
        use_m1_operator=True,    # Sử dụng toán tử mutation m1
        use_m2_operator=True,    # Sử dụng toán tử mutation m2
        num_samplers=1,          # Số thread sampling
        num_evaluators=4,        # Số thread evaluation (parallel)
        num_objs=2,              # Số mục tiêu (có thể điều chỉnh)
        multi_thread_or_process_eval='thread'  # 'thread' hoặc 'process'
    )    
    # Chạy thuật toán
    method.run()
    
    print("Thí nghiệm MEoH-VRPTW hoàn thành!")
    print("Kết quả được lưu tại: logs/meoh_vrptw")