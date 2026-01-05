## Stage 1: Assess Lint Failures
**Goal**: Identify all ruff violations and locate corresponding code patterns.
**Success Criteria**: Every reported rule is mapped to a concrete change location.
**Tests**: ruff (pre-commit hook output)
**Status**: Complete
- [x] Review files flagged by ruff and capture needed edits
- [x] Locate similar patterns in codebase to match conventions

## Stage 2: Apply Minimal Fixes
**Goal**: Implement the smallest edits that satisfy ruff without behavior changes.
**Success Criteria**: All listed ruff errors are resolved.
**Tests**: ruff (pre-commit hook output)
**Status**: Complete
- [x] Fix string ambiguity and whitespace-only lines
- [x] Replace open() with Path.open()
- [x] Simplify conditional blocks flagged by SIM rules
- [x] Remove unused variable/duplicate fields
- [x] Use timezone-aware datetime
- [x] Resolve undefined TrainingConfig type
- [x] Remove unnecessary assignments before return

## Stage 3: Verify
**Goal**: Confirm lint passes and update documentation.
**Success Criteria**: ruff hook succeeds; task checklist updated.
**Tests**: ruff (pre-commit hook output)
**Status**: Complete
- [x] Run or re-run lint hook if available
- [x] Mark tasks complete in this file
