"""The two-line ordered-crossing rule, and the `side()` primitive every counter here builds on.

The rule
--------
A person is counted as **boarding** when their tracked point crosses the ENTRANCE line and then,
in that order, crosses the CABIN line; as **alighting** when they cross CABIN and then ENTRANCE.
Nothing counts until both crossings have happened, in order, on the same track.

Direction therefore comes from the ORDER of two crossings, not from the sign of one. Someone who
hovers on a boundary jittering back and forth crosses the same line twice and completes no pair;
someone who leans in to talk to the driver and withdraws is never counted.

A crossing only counts inside the segment's own x-span (widened by a pad in
`count_two_line_v3.crossing`). Without that, the infinite line would be tripped by people moving
about the cabin far from the doorway.

Later versions change the tracked point (the pose "ladder", then the upper body) and the second
test; `side()` is unchanged throughout.
"""


def side(line, pt):
    """Signed side of `pt` relative to the directed line. Positive is the vehicle interior.

    Which side is positive is decided by the order the line's endpoints are written in, so check
    it by substituting a point known to be inside. A reversed line gives a silent all-zero run.
    """
    (px, py), (qx, qy) = line
    return (qx - px) * (pt[1] - py) - (qy - py) * (pt[0] - px)
