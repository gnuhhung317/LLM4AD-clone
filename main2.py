from llm4ad.task.optimization.vrptw_construct_3O import VRPTWEvaluation3O
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.meoh import MEoH, MEoHProfiler

if __name__ == '__main__':
    llm = HttpsApi(
    )
    
    # Sử dụng lớp đánh giá 3 mục tiêu mới
    task = VRPTWEvaluation3O()
    
    # Cấu hình MEoH cho tối ưu 3 mục tiêu
    method = MEoH(
        llm=llm,
        evaluation=task,
        profiler=MEoHProfiler(
            num_objs=3,
            log_dir='logs/meoh_vrptw_3obj',
            log_style='complex',
            evaluation_name='VRPTW-3Obj',
            method_name='MEoH'        ),
        max_generations=10,      # Số thế hệ tối đa
        max_sample_nums=10*20,   # max_generations * pop_size = 200
        pop_size=20,             # Kích thước population
        selection_num=5,         # Số cá thể được chọn trong crossover
        use_e2_operator=True,    # Sử dụng toán tử evolution e2
        use_m1_operator=True,    # Sử dụng toán tử mutation m1
        use_m2_operator=True,    # Sử dụng toán tử mutation m2
        num_samplers=1,          # Số thread sampling
        num_evaluators=4,        # Số thread evaluation (parallel)
        num_objs=3,              # Số mục tiêu (3 objectives)
        debug_mode=False,        # Bật debug để xem chi tiết
        multi_thread_or_process_eval='thread'  # 'thread' hoặc 'process'
    )
    
    # Chạy thuật toán
    method.run()
    
    print("Thí nghiệm MEoH-VRPTW với 3 mục tiêu hoàn thành!")
    print("Mục tiêu 1: Tổng khoảng cách (Total Distance)")
    print("Mục tiêu 2: Vi phạm thời gian (Time Violations)")
    print("Mục tiêu 3: Thời gian thực thi thuật toán (Execution Time)")
    print("Kết quả được lưu tại: logs/meoh_vrptw_3obj")
