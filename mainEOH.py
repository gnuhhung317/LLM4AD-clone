from llm4ad.task.optimization.vrptw_construct_3O import Evaluation4Violations
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH
from llm4ad.method.meoh import MEoHProfiler

if __name__ == '__main__':
    llm = HttpsApi(
    )
    
    # Sử dụng lớp đánh giá 3 mục tiêu (mặc dù chỉ tối ưu 1 mục tiêu chính)
    task = Evaluation4Violations()
    
    # Cấu hình EoH cho tối ưu 3 mục tiêu
    method = EoH(
        llm=llm,
        evaluation=task,
        profiler=MEoHProfiler(
            num_objs=3,
            log_dir='logs/eoh_vrptw_1obj_3criteria',
            log_style='complex',
            evaluation_name='Evaluation4Violations',
            method_name='eoH'        ),
        max_generations=10,      # Số thế hệ tối đa
        max_sample_nums=10 * 20,   # max_generations * pop_size = 200
        pop_size=40,             # Kích thước population
        selection_num=5,         # Số cá thể được chọn trong crossover
        use_e2_operator=True,    # Sử dụng toán tử evolution e2
        use_m1_operator=True,    # Sử dụng toán tử mutation m1
        use_m2_operator=True,    # Sử dụng toán tử mutation m2
        num_samplers=1,          # Số thread sampling
        num_evaluators=1,        # Số thread evaluation (parallel)
        debug_mode=False,        # Bật debug để xem chi tiết
        multi_thread_or_process_eval='thread'  # 'thread' hoặc 'process'
    )

    # Chạy thuật toán
    method.run()
    
    print("Thí nghiệm EoH-VRPTW với 1 mục tiêu hoàn thành!")
    print("Mục tiêu 1: Vi phạm thời gian (Time Violations)")
    print("Tiêu chí quan sát 1: Tổng khoảng cách (Total Distance)")
    print("Tiêu chí quan sát 2: Thời gian thực thi thuật toán (Execution Time)")
    print("Kết quả được lưu tại: logs/eoh_vrptw_3obj")