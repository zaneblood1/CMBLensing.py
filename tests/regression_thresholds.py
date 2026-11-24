#Every test file lists the Python v.s. Julia differences measured when its thresholds were set (2026-11-24).
#A comparison fails once its difference grows past REGRESSION_MARGIN times that measured value.

#how many times its measured difference a comparison may reach before the test fails
REGRESSION_MARGIN = 2
#allowed difference for the comparisons that agree exactly or to round-off
ROUND_OFF_TOLERANCE = 1e-12

def max_allowed(measured_difference):
    """The largest difference a comparison may show, given the difference measured when the test was written."""
    return max(REGRESSION_MARGIN * measured_difference, ROUND_OFF_TOLERANCE)
