"""Per-seed recovery metrics from unsmoothed greedy evaluation points."""


def summarize_recovery(evaluations, failure_start, budget, pre_window=300000,
                       nominal_failure_t=None):
    """Return J_pre, J0 and AUC, without extrapolating missing budget endpoints."""
    nominal_failure_t = failure_start if nominal_failure_t is None else nominal_failure_t
    pre = [item["win_rate"] for item in evaluations
           if nominal_failure_t - pre_window <= item["t_env"] < nominal_failure_t
           and not item["failure_active"]]
    post = sorted((item["t_env"] - failure_start, item["win_rate"])
                  for item in evaluations
                  if item["failure_active"] and item["t_env"] > failure_start)
    result = {
        "J_pre": sum(pre) / len(pre) if pre else None,
        "J0": post[0][1] if post else None,
        "b1": post[0][0] if post else None,
        "AUC_rec": None,
        "AUC_gain": None,
    }
    if len(post) < 2 or post[0][0] >= budget:
        return result
    points = [(step, win) for step, win in post if step <= budget]
    if not points:
        return result
    if points[-1][0] < budget:
        upper = next(((step, win) for step, win in post if step > budget), None)
        if upper is None:
            return result
        low_step, low_win = points[-1]
        high_step, high_win = upper
        interpolated = low_win + (high_win - low_win) * (budget - low_step) / (high_step - low_step)
        points.append((budget, interpolated))
    area = sum((step2 - step1) * (win1 + win2) / 2
               for (step1, win1), (step2, win2) in zip(points, points[1:]))
    result["AUC_rec"] = area / (budget - points[0][0])
    result["AUC_gain"] = result["AUC_rec"] - result["J0"]
    return result
