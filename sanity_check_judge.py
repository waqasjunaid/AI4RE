"""
sanity_check_judge.py
------------------------
Negative-control test: sends a handful of DELIBERATELY faithful and
DELIBERATELY unfaithful (source, statement) pairs directly to the judge
model, to confirm it can actually distinguish them -- not just rubber-
stamping everything as SUPPORTED (the same failure mode documented for
llama3.1:70b's fixed 4/6 self-judging in Section 8.2 of the paper).

If the judge model gets the obviously-wrong cases wrong too, that's a
serious problem worth knowing about BEFORE reporting a clean 100%
faithfulness result from it.

Usage:
    python sanity_check_judge.py --judge-model mistral:7b
"""
import argparse
import sys
sys.path.insert(0, ".")
from faithfulness_check import call_judge

CASES = [
    # (source, statement, expected_category, why)
    ("The system shall lock a user account after 5 consecutive failed login attempts.",
     "The system shall lock a user account after repeated failed login attempts.",
     "SUPPORTED", "faithful paraphrase, no invented specifics"),

    ("The system shall lock a user account after 5 consecutive failed login attempts.",
     "The system shall lock a user account after exactly 3 failed login attempts within 10 seconds.",
     "PARTIALLY_SUPPORTED_OR_UNSUPPORTED",
     "invents a DIFFERENT number (3 vs 5) and a time window not in source -- should NOT be plain SUPPORTED"),

    ("All tokens must be signed with RS256 (asymmetric).",
     "All tokens must be signed with RS256.",
     "SUPPORTED", "faithful, drops a parenthetical detail but keeps the core claim"),

    ("All tokens must be signed with RS256 (asymmetric).",
     "All tokens must be encrypted using quantum-resistant lattice cryptography.",
     "PARTIALLY_SUPPORTED_OR_UNSUPPORTED",
     "completely different, unrelated claim -- should NOT be SUPPORTED"),

    ("The system shall provide a password reset mechanism via email.",
     "The system shall allow users to delete their entire account and all associated data within 24 hours of a support request.",
     "PARTIALLY_SUPPORTED_OR_UNSUPPORTED",
     "totally unrelated feature -- should NOT be SUPPORTED"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", required=True)
    args = ap.parse_args()

    print("Sanity-checking judge model: {}\n".format(args.judge_model))
    n_correct = 0
    for i, (source, statement, expected, why) in enumerate(CASES, 1):
        verdict = call_judge(args.judge_model, source, statement)
        if expected == "SUPPORTED":
            ok = verdict == "SUPPORTED"
        else:
            ok = verdict in ("PARTIALLY_SUPPORTED", "UNSUPPORTED")
        n_correct += int(ok)
        status = "OK" if ok else "**WRONG**"
        print("[{}] {} (expected: {})".format(i, status, expected))
        print("    source:    {}".format(source))
        print("    statement: {}".format(statement))
        print("    why:       {}".format(why))
        print("    judge said: {}\n".format(verdict))

    print("=" * 60)
    print("Sanity check: {}/{} cases correct".format(n_correct, len(CASES)))
    if n_correct < len(CASES):
        print("\nWARNING: the judge model failed to flag at least one obviously")
        print("unfaithful statement. Treat its SUPPORTED verdicts on your real")
        print("TR sample with caution -- it may be rubber-stamping, similar to")
        print("the fixed-output bias documented for llama3.1:70b in Section 8.2.")
    else:
        print("\nJudge model correctly distinguished all faithful/unfaithful cases.")
        print("A high SUPPORTED rate on your real TR sample is more trustworthy")
        print("given this passed.")


if __name__ == "__main__":
    main()
