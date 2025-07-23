#!/usr/bin/env python3
import subprocess
import ast

scores = []

# The literal prefix to locate result_type
RT_KEY = "result_type='"

for i in range(1, 101):
    try:
        # 1) Run your simulator
        subprocess.run(
            ["python3", "match_simulator.py",
             "--submissions", "4:refactor.py",
             "--engine"],
            check=True
        )

        # 2) Read the log
        with open("output/engine.log", "r") as f:
            lines = f.readlines()

        # 3) Grab the last non-empty line
        for raw in reversed(lines):
            line = raw.strip()
            if line:
                score_line = line
                break
        else:
            # empty log
            continue

        # 4) Direct char‑position check for SUCCESS
        rt_index = score_line.find(RT_KEY)
        if rt_index == -1:
            continue
        # check the very next char after the opening quote
        if score_line[rt_index + len(RT_KEY)] != "S":
            continue

        # 5) Extract the score dict via brace‑counting
        idx = score_line.find("score=", rt_index)
        if idx == -1:
            continue
        brace_start = score_line.find("{", idx)
        if brace_start == -1:
            continue

        depth = 0
        for j, ch in enumerate(score_line[brace_start:], start=brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    score_str = score_line[brace_start : j+1]
                    break
        else:
            # unmatched braces
            continue

        # 6) Parse and store the highest score
        score_dict = ast.literal_eval(score_str)
        scores.append(max(score_dict.values()))

    except subprocess.CalledProcessError:
        # simulator itself crashed; skip this run
        continue
    except Exception:
        # any parsing error; skip too
        continue

    # 7) Every 100 iterations, print running stats
    if scores:
        running_avg = sum(scores) / len(scores)
        print(f"After {i} runs: {len(scores)} successes, running avg = {running_avg:.2f}")
    else:
        print(f"After {i} runs: no successful scores yet")

# 8) Final summary
if scores:
    overall_avg = sum(scores) / len(scores)
    print(f"\nDone: {len(scores)} successful runs, overall average = {overall_avg:.2f}")
else:
    print("Done: no successful runs recorded.")