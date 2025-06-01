#!/usr/bin/env python3

# Test simple class definition
import sys
sys.path.insert(0, '.')

try:
    from llm4ad.base import Evaluation
    print("✓ Successfully imported Evaluation")
except Exception as e:
    print(f"✗ Failed to import Evaluation: {e}")

try:
    from llm4ad.task.optimization.vrptw_construct_3O.get_instance import GetData
    print("✓ Successfully imported GetData")
except Exception as e:
    print(f"✗ Failed to import GetData: {e}")

try:
    from llm4ad.task.optimization.vrptw_construct_3O.template import template_program, task_description
    print("✓ Successfully imported template")
except Exception as e:
    print(f"✗ Failed to import template: {e}")

# Test if we can create a simple class that inherits from Evaluation
try:
    class SimpleTest(Evaluation):
        def __init__(self):
            super().__init__(
                template_program="test",
                task_description="test",
                use_numba_accelerate=False,
                timeout_seconds=30
            )
    
    print("✓ Successfully created SimpleTest class")
    instance = SimpleTest()
    print("✓ Successfully instantiated SimpleTest")
except Exception as e:
    print(f"✗ Failed to create SimpleTest: {e}")
