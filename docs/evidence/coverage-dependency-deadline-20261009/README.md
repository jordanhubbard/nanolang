# My coverage dependency setup budget

I retain job113780446789 at 54024f19e in hosted.log. All three 180-second attempts exit137 before compilation, with package downloads continuing during the attempts. This establishes deadline exhaustion, not its underlying network cause.

I give the coverage dependency step two 420-second attempts with a five-second backoff. Their 845-second nominal total fits a 900-second step limit. I increase the job limit from 45 to 50 minutes so setup still leaves the previous 35-minute budget for the remaining steps. I retain all required packages, compiler checks, sanitizer settings and coverage thresholds.

All five workflow tests and the helper retry/exhaustion controls pass. I retain both logs. Fresh hosted completion is still required under #982 and #976; these local controls do not establish package download success or release readiness.
