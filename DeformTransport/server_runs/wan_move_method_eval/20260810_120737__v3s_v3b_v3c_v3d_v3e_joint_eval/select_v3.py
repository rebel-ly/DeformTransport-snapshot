import json
import sys
from pathlib import Path


CANDIDATES = [
    "v3s",
    "v3b",
    "v3c",
    "v3d",
    "v3e",
]

CASES = [
    "santa",
    "tree",
]


out = Path(
    sys.argv[
        1
    ]
)

appearance = json.loads(
    (
        out
        / "appearance_report.json"
    ).read_text()
)

motion = {
    case:
        json.loads(
            (
                out
                / f"{case}_motion_report.json"
            ).read_text()
        )
    for case in CASES
}


rows = []


for candidate in CANDIDATES:

    motion_status = {
        case:
            motion[
                case
            ][
                "vs_old_correct"
            ][
                candidate
            ][
                "motion_safety"
            ]
        for case in CASES
    }

    appearance_status = {
        case:
            appearance[
                "cases"
            ][
                case
            ][
                "vs_old_correct"
            ][
                candidate
            ][
                "status"
            ]
        for case in CASES
    }

    motion_safe = all(
        status == "PASS"
        for status in (
            motion_status.values()
        )
    )

    appearance_regression = any(
        status == "REGRESS"
        for status in (
            appearance_status.values()
        )
    )

    improve_count = sum(
        status == "IMPROVE"
        for status in (
            appearance_status.values()
        )
    )

    tie_count = sum(
        status == "TIE"
        for status in (
            appearance_status.values()
        )
    )

    if (
        not motion_safe
        or
        appearance_regression
    ):

        tier = -1
        eligible = False

    elif improve_count == 2:

        tier = 3
        eligible = True

    elif (
        improve_count == 1
        and
        tie_count == 1
    ):

        tier = 2
        eligible = True

    elif tie_count == 2:

        tier = 1
        eligible = True

    else:

        tier = -1
        eligible = False

    relative = [
        appearance[
            "cases"
        ][
            case
        ][
            "vs_old_correct"
        ][
            candidate
        ][
            "relative_mean_improvement"
        ]
        for case in CASES
    ]

    row = {
        "candidate":
            candidate,

        "motion_safety":
            motion_status,

        "appearance_status":
            appearance_status,

        "eligible":
            eligible,

        "tier":
            tier,

        "mean_relative_tc_mar_lab_improvement":
            sum(
                relative
            )
            / 2.0,

        "appearance": {
            case: {
                "old_correct":
                    appearance[
                        "cases"
                    ][
                        case
                    ][
                        "methods"
                    ][
                        "old_correct"
                    ][
                        "tc_mar_lab"
                    ][
                        "mean"
                    ],

                "candidate":
                    appearance[
                        "cases"
                    ][
                        case
                    ][
                        "methods"
                    ][
                        candidate
                    ][
                        "tc_mar_lab"
                    ][
                        "mean"
                    ],

                "old_minus_candidate_ci":
                    appearance[
                        "cases"
                    ][
                        case
                    ][
                        "vs_old_correct"
                    ][
                        candidate
                    ][
                        "bootstrap_95_ci"
                    ],

                "vs_rw":
                    appearance[
                        "cases"
                    ][
                        case
                    ][
                        "vs_realwonder"
                    ][
                        candidate
                    ][
                        "decision"
                    ],
            }

            for case in CASES
        },

        "motion": {
            case: {
                "old_correct":
                    motion[
                        case
                    ][
                        "methods"
                    ][
                        "old_correct"
                    ][
                        "transition_mean_epe_mean"
                    ],

                "candidate":
                    motion[
                        case
                    ][
                        "methods"
                    ][
                        candidate
                    ][
                        "transition_mean_epe_mean"
                    ],

                "candidate_minus_old_ci":
                    motion[
                        case
                    ][
                        "vs_old_correct"
                    ][
                        candidate
                    ][
                        "bootstrap_95_ci"
                    ],

                "vs_rw":
                    motion[
                        case
                    ][
                        "vs_realwonder"
                    ][
                        candidate
                    ][
                        "decision"
                    ],
            }

            for case in CASES
        },
    }

    rows.append(
        row
    )


eligible = [
    row
    for row in rows
    if row[
        "eligible"
    ]
]


if eligible:

    winner = max(
        eligible,
        key=lambda row: (
            row[
                "tier"
            ],
            row[
                "mean_relative_tc_mar_lab_improvement"
            ],
        ),
    )[
        "candidate"
    ]

    decision = (
        "WINNER_SELECTED"
    )

else:

    winner = None
    decision = (
        "NO_WINNER"
    )


report = {
    "decision":
        decision,

    "winner":
        winner,

    "selection_basis":
        "Santa+Tree development only; "
        "motion safety -> "
        "TC-MAR Lab significance tier -> "
        "mean relative TC-MAR Lab "
        "improvement tie-break",

    "sandhouse_used_for_selection":
        False,

    "candidates":
        rows,
}


(
    out
    / "selection_report.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    )
    + "\n"
)


with (
    out
    / "selection_summary.tsv"
).open(
    "w"
) as f:

    f.write(
        "candidate\t"
        "motion_santa\t"
        "motion_tree\t"
        "app_santa\t"
        "app_tree\t"
        "eligible\t"
        "tier\t"
        "mean_rel_tc_mar\n"
    )

    for row in rows:

        f.write(
            f"{row['candidate']}\t"
            f"{row['motion_safety']['santa']}\t"
            f"{row['motion_safety']['tree']}\t"
            f"{row['appearance_status']['santa']}\t"
            f"{row['appearance_status']['tree']}\t"
            f"{row['eligible']}\t"
            f"{row['tier']}\t"
            f"{row['mean_relative_tc_mar_lab_improvement']:.8f}\n"
        )


print(
    "=== FROZEN V3 SELECTION ==="
)

for row in rows:

    print(
        row[
            "candidate"
        ],

        "motion",
        row[
            "motion_safety"
        ],

        "appearance",
        row[
            "appearance_status"
        ],

        "eligible",
        row[
            "eligible"
        ],

        "tier",
        row[
            "tier"
        ],

        "mean_rel",
        f"{row['mean_relative_tc_mar_lab_improvement']:.6f}",
    )


print(
    "DECISION=",
    decision,
)

print(
    "WINNER=",
    winner,
)
