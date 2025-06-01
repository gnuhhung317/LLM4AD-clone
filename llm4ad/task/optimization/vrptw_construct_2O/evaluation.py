# Module Name: VRPTWEvaluation2O
# Last Revision: 2025/5/27
# Description: Multi-objective evaluation for the Vehicle Routing Problem with Time Windows (VRPTW).
#       This module evaluates VRPTW solutions based on two objectives:
#       1. Total travel distance (to be minimized)
#       2. Total time window violations/waiting time (to be minimized)
#       This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#   - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#   - problem_size: Number of customers to serve (excluding the depot): int (default: 50).
#   - n_instance: Number of problem instances to generate: int (default: 16).
# 
# References:
#   - Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
#       Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
#       with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Any
import copy
import numpy as np
from llm4ad.base import Evaluation
from .get_instance import GetData
from .template import template_program, task_description


class VRPTWEvaluation2O(Evaluation):
    def __init__(self,
                 timeout_seconds=30,
                 problem_size=50,
                 n_instance=16,
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.timeout_seconds = timeout_seconds
        self.problem_size = problem_size
        self.n_instance = n_instance

        getData = GetData(self.n_instance, self.problem_size + 1)
        self._datasets = getData.generate_instances()

    def tour_cost_and_violations(self, distance_matrix, solution, time_service, time_windows):
        """Calculate both total distance and time violations for a solution."""
        total_distance = 0
        total_violations = 0
        current_time = 0

        for j in range(len(solution) - 1):
            travel_time = distance_matrix[int(solution[j]), int(solution[j + 1])]
            current_time += travel_time

            # Check time window constraints
            arrival_time = current_time
            earliest_time = time_windows[solution[j + 1]][0]
            latest_time = time_windows[solution[j + 1]][1]

            # Calculate waiting time (if arrived too early)
            waiting_time = max(0, earliest_time - arrival_time)
            current_time = max(arrival_time, earliest_time)

            # Calculate violation (if arrived too late)
            violation = max(0, arrival_time - latest_time)
            
            # Add to totals
            total_distance += travel_time
            total_violations += waiting_time + violation

            # Add service time
            current_time += time_service[solution[j + 1]]
            
            # Reset time when returning to depot
            if solution[j + 1] == 0:
                current_time = 0

        return total_distance, total_violations

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, heuristic):
        distances = np.ones(self.n_instance)
        violations = np.ones(self.n_instance)
        n_ins = 0

        for instance, distance_matrix, demands, vehicle_capacity, time_service, time_windows in self._datasets:
            route = []
            current_load = 0
            current_node = 0
            current_time = 0
            route.append(current_node)
            unvisited_nodes = set(range(1, self.problem_size + 1))  # Assuming node 0 is the depot
            all_nodes = np.array(list(unvisited_nodes))
            feasible_unvisited_nodes = all_nodes

            unvisited_nodes_depot = np.array(list(unvisited_nodes))

            while unvisited_nodes:
                next_node = heuristic(current_node,
                                      0,
                                      feasible_unvisited_nodes,
                                      vehicle_capacity - current_load,
                                      current_time,
                                      copy.deepcopy(demands),
                                      copy.deepcopy(distance_matrix),
                                      copy.deepcopy(time_windows))
                
                if next_node == 0:
                    route.append(next_node)
                    current_load = 0
                    current_time = 0
                    current_node = 0
                    unvisited_nodes_depot = np.array(list(unvisited_nodes))
                else:
                    travel_time = distance_matrix[current_node, next_node]
                    current_time += travel_time

                    if current_time < time_windows[next_node][0]:
                        current_time = time_windows[next_node][0]
                    if max(current_time, time_windows[next_node][0]) > time_windows[next_node][1]:
                        return None  # Infeasible solution

                    current_time += time_service[next_node]
                    current_load += demands[next_node]
                    current_node = next_node
                    route.append(current_node)
                    unvisited_nodes.remove(next_node)

                # Update feasible nodes
                feasible_unvisited_nodes = []
                for node in unvisited_nodes:
                    travel_time_to_node = distance_matrix[current_node, node]
                    arrival_time = current_time + travel_time_to_node
                    
                    # Check capacity and time window constraints
                    if (demands[node] <= vehicle_capacity - current_load and
                        arrival_time <= time_windows[node][1]):
                        feasible_unvisited_nodes.append(node)

                # If no feasible nodes and there are still unvisited nodes, return to depot
                if not feasible_unvisited_nodes and unvisited_nodes:
                    feasible_unvisited_nodes = [0]
                
                feasible_unvisited_nodes = np.array(feasible_unvisited_nodes)

            # Ensure route returns to depot at the end
            if route[-1] != 0:
                route.append(0)

            # Check if all nodes were visited
            if len(set(route)) != self.problem_size + 1:
                return None

            # Calculate both objectives
            total_distance, total_violations = self.tour_cost_and_violations(
                distance_matrix, route, time_service, time_windows)
            
            distances[n_ins] = total_distance
            violations[n_ins] = total_violations

            n_ins += 1
            if n_ins == self.n_instance:
                break

        # Return both objectives as negative values (for maximization in MEoH)
        avg_distance = np.average(distances)
        avg_violations = np.average(violations)
        
        # Return as numpy array with two objectives
        return np.array([-avg_distance, -avg_violations])


if __name__ == '__main__':
    def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray, current_time: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray, time_windows: np.ndarray) -> int:
        """Design a novel algorithm to select the next node in each step.
        Args:
            current_node: ID of the current node.
            depot: ID of the depot.
            unvisited_nodes: Array of IDs of unvisited nodes.
            rest_capacity: Rest capacity of vehicle
            current_time: Current time
            demands: Demands of nodes
            distance_matrix: Distance matrix of nodes.
            time_windows: Time windows of nodes.
        Return:
            ID of the next node to visit.
        """
        best_node = -1
        best_value = -float('inf')

        for node in unvisited_nodes:
            if demands[node] <= rest_capacity:
                travel_time = distance_matrix[current_node, node]
                arrival_time = current_time + travel_time

                if arrival_time <= time_windows[node][1]:  # Checking if within time window
                    wait_time = max(0, time_windows[node][0] - arrival_time)
                    effective_time = arrival_time + wait_time
                    distance_to_demand_ratio = travel_time / demands[node] if demands[node] > 0 else float('inf')

                    if distance_to_demand_ratio > best_value:
                        best_value = distance_to_demand_ratio
                        best_node = node

        return best_node if best_node != -1 else depot


    eval = VRPTWEvaluation2O()
    res = eval.evaluate_program('', select_next_node)
    print(f"Multi-objective result: {res}")
    print(f"Objective 1 (negative distance): {res[0]}")
    print(f"Objective 2 (negative violations): {res[1]}")
