# CSC242: Project 3 - Uncertainty Modeling
# Date: April 3, 2025

import sys
import random
import math
import os

# KL Divergence (for experiments)
def kl(p, q):
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)

# Parse Bayesian network file
def parse_network(filename):
    try:
        with open(filename) as f:
            # Read all non-empty, non-comment lines
            lines = []
            for line in f:
                line = line.split('#')[0].strip() 
                if line:  
                    lines.append(line)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return [], {}, {}

    try:
        i = 0
        num_vars = int(lines[i])
        i += 1
        
        vars = []
        domains = {}
        for _ in range(num_vars):
            parts = lines[i].split()
            vars.append(parts[0])
            domains[parts[0]] = parts[1:]
            i += 1

        num_cpts = int(lines[i])
        i += 1
        cpts = {}

        for _ in range(num_cpts):
            while i < len(lines) and not lines[i]:
                i += 1
            if i >= len(lines):
                break

            header = lines[i].split()
            i += 1
            child = header[0]
            parents = header[1:] if len(header) > 1 else []

            table = []
            while i < len(lines) and lines[i] and not lines[i].startswith('#'):
                if all(part.replace('.', '').isdigit() for part in lines[i].split()):
                    table.append(list(map(float, lines[i].split())))
                    i += 1
                else:
                    break

            cpts[child] = {'parents': parents, 'table': table}

        return vars, domains, cpts

    except Exception as e:
        print(f"Error parsing line {i+1}: '{lines[i] if i < len(lines) else 'EOF'}'")
        print(f"Full error: {str(e)}")
        return [], {}, {}

# Compute joint probability
def joint_prob(assignment, vars, domains, cpts):
    p = 1.0
    for var in vars:
        if var not in assignment:
            return 0.0
        val = assignment[var]
        val_idx = domains[var].index(val)
        parents = cpts[var]['parents']
        table = cpts[var]['table']

        if not parents:
            probs = table[0]
        else:
            index = 0
            stride = 1
            for pvar in reversed(parents):
                index += domains[pvar].index(assignment[pvar]) * stride
                stride *= len(domains[pvar])
            probs = table[index]

        p *= probs[val_idx]
    return p

# Exact inference
def exact_inference(query, evidence, vars, domains, cpts):
    if query not in vars:
        print("Unknown query variable.")
        return []
    hidden = [v for v in vars if v != query and v not in evidence]
    counts = [0.0] * len(domains[query])

    def recur(i, partial):
        if i == len(hidden):
            for j, val in enumerate(domains[query]):
                partial[query] = val
                counts[j] += joint_prob(partial, vars, domains, cpts)
        else:
            var = hidden[i]
            for val in domains[var]:
                partial[var] = val
                recur(i+1, partial)

    recur(0, dict(evidence))
    total = sum(counts)
    return [round(x/total, 4) for x in counts] if total > 0 else [0.0] * len(counts)

# Rejection sampling
def rejection_sample(query, evidence, vars, domains, cpts, N):
    counts = [0.0] * len(domains[query])
    for _ in range(N):
        sample = {}
        consistent = True
        for var in vars:
            parents = cpts[var]['parents']
            table = cpts[var]['table']

            if not parents:
                probs = table[0]
            else:
                index = 0
                stride = 1
                for p in reversed(parents):
                    if p not in sample:
                        consistent = False
                        break
                    index += domains[p].index(sample[p]) * stride
                    stride *= len(domains[p])
                if not consistent:
                    break
                probs = table[index]

            r = random.random()
            cum = 0.0
            for i, prob in enumerate(probs):
                cum += prob
                if r <= cum:
                    sample[var] = domains[var][i]
                    break

        if consistent and all(sample.get(k) == v for k, v in evidence.items()):
            idx = domains[query].index(sample[query])
            counts[idx] += 1

    total = sum(counts)
    return [round(x/total, 4) if total > 0 else 0.0 for x in counts]

# Gibbs sampling
def gibbs_sample(query, evidence, vars, domains, cpts, N, burn=100):
    state = dict(evidence)
    hidden = [v for v in vars if v not in evidence]
    for var in hidden:
        state[var] = random.choice(domains[var])

    counts = [0.0] * len(domains[query])
    for i in range(N + burn):
        for var in hidden:
            dist = []
            for val in domains[var]:
                state[var] = val
                dist.append(joint_prob(state, vars, domains, cpts))
            total = sum(dist)
            r = random.random()
            acc = 0
            for j, val in enumerate(domains[var]):
                acc += dist[j] / total
                if r <= acc:
                    state[var] = val
                    break
        if i >= burn:
            idx = domains[query].index(state[query])
            counts[idx] += 1

    total = sum(counts)
    return [round(x/total, 4) if total > 0 else 0.0 for x in counts]

# Read vector from txt file
def load_vector(filename):
    with open(filename) as f:
        return list(map(float, f.read().strip().split()))

# Save results to file for report/logging
def save_result(query, method, sample_count, result, kl_val=None):
    with open("results.csv", "a") as f:
        f.write(f"{query},{method},{sample_count},{result},{kl_val if kl_val is not None else ''}\n")

# REPL interface
def repl():
    vars, domains, cpts = [], {}, {}
    print("BayesNet REPL started. Type 'help' for commands.")
    while True:
        try:
            line = input("BN > ").strip()
            if not line:
                continue
            if line == 'quit':
                break
            elif line.startswith('load'):
                fname = line.split(maxsplit=1)[1]
                vars, domains, cpts = parse_network(fname)
                if vars:
                    print("Network loaded.")
            elif line.startswith(('xquery', 'rquery', 'gquery')):
                parts = line.split('|')
                left = parts[0].split()
                query = left[1]
                count = int(left[2]) if len(left) > 2 else 1000
                evidence = {}
                if len(parts) > 1:
                    for pair in parts[1].strip().split():
                        k, v = pair.split('=')
                        evidence[k] = v
                if query not in vars:
                    print("Unknown query variable.")
                    continue
                if line.startswith('xquery'):
                    result = exact_inference(query, evidence, vars, domains, cpts)
                    method = 'exact'
                elif line.startswith('rquery'):
                    result = rejection_sample(query, evidence, vars, domains, cpts, count)
                    method = 'rejection'
                else:
                    result = gibbs_sample(query, evidence, vars, domains, cpts, count)
                    method = 'gibbs'
                print(result, "\n")
                save_result(query, method, count, result)
            elif line.startswith('compare'):
                parts = line.split('with')
                query_part = parts[0].strip().split('|')
                method_and_var = query_part[0].strip().split()
                method, query = method_and_var[1], method_and_var[2]
                evidence = {}
                if len(query_part) > 1:
                    for pair in query_part[1].strip().split():
                        k, v = pair.split('=')
                        evidence[k] = v
                filename = parts[1].strip()
                ref = load_vector(filename)
                if method == 'x':
                    result = exact_inference(query, evidence, vars, domains, cpts)
                    method_name = 'exact'
                elif method == 'r':
                    result = rejection_sample(query, evidence, vars, domains, cpts, 1000)
                    method_name = 'rejection'
                else:
                    result = gibbs_sample(query, evidence, vars, domains, cpts, 1000)
                    method_name = 'gibbs'
                kl_val = round(kl(ref, result), 5)
                print("Result:", result)
                print("KL divergence:", kl_val, "\n")
                save_result(query, method_name, 1000, result, kl_val)
            elif line == 'help':
                print("Commands:")
                print("  load FILE")
                print("  xquery QUERY [SAMPLES] | E1=v1 E2=v2 ...")
                print("  rquery QUERY [SAMPLES] | E1=v1 ...")
                print("  gquery QUERY [SAMPLES] | E1=v1 ...")
                print("  compare METHOD QUERY | E1=v1 ... with FILE")
                print("  quit")
            else:
                print("Unknown command\n")
        except Exception as e:
            print("Error:", e, "\n")

if __name__ == '__main__':
    repl()