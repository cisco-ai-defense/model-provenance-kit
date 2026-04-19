# Pull Request

## Description

Provide a clear and concise description of the changes in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test coverage improvement

## Related Issues

Closes #[issue number]
Fixes #[issue number]
Related to #[issue number]

## Changes Made

- Change 1: [describe]
- Change 2: [describe]
- Change 3: [describe]

## Testing

### Test Coverage

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Test coverage maintained or improved

### Manual Testing

Describe manual testing performed:

```bash
# Run unit tests
uv run pytest -m "not slow" --tb=short -q

# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/
```

**Results:**
- Expected: [describe]
- Actual: [describe]

## Checklist

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Type hints added where applicable
- [ ] Docstrings added/updated for public APIs
- [ ] No hardcoded credentials or secrets
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate

### Documentation
- [ ] README updated (if needed)
- [ ] API documentation updated (if needed)
- [ ] Code comments added for complex logic

### Security
- [ ] No new security vulnerabilities introduced
- [ ] Input validation added where needed
- [ ] Follows security best practices from workspace rules
- [ ] No eval/exec on user input without sanitization

### Testing
- [ ] Tests pass: `uv run pytest -m "not slow" --tb=short -q`
- [ ] Lint passes: `uv run ruff check src/ tests/`
- [ ] Type check passes: `uv run mypy src/`
- [ ] No regressions in existing functionality
- [ ] Edge cases covered

## Performance Impact

- [ ] No significant performance regression
- [ ] Performance benchmarks run (if applicable)
- [ ] Resource usage is acceptable

## Screenshots (if applicable)

Add screenshots or output examples if relevant.

## Additional Notes

Any additional information reviewers should know.

## Reviewer Checklist

For reviewers:
- [ ] Code changes are clear and well-documented
- [ ] Tests are comprehensive
- [ ] No security issues introduced
- [ ] Performance is acceptable
- [ ] Documentation is updated
