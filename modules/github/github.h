#ifndef NANOLANG_GITHUB_H
#define NANOLANG_GITHUB_H

#include <stdint.h>

// GitHub API client functions
// All functions return JSON strings or error messages
// Authentication uses GITHUB_TOKEN environment variable or explicit token parameter

// Core API functions
char* nl_github_get_repo(const char* owner, const char* repo, const char* token);
char* nl_github_list_issues(const char* owner, const char* repo, const char* state, const char* token);
char* nl_github_get_issue(const char* owner, const char* repo, int64_t issue_number, const char* token);
char* nl_github_create_issue(const char* owner, const char* repo, const char* title, const char* body, const char* token);
char* nl_github_update_issue(const char* owner, const char* repo, int64_t issue_number, const char* title, const char* body, const char* state, const char* token);
char* nl_github_create_issue_comment(const char* owner, const char* repo, int64_t issue_number, const char* body, const char* token);
char* nl_github_list_prs(const char* owner, const char* repo, const char* state, const char* token);
char* nl_github_get_pr(const char* owner, const char* repo, int64_t pr_number, const char* token);
char* nl_github_create_pr(const char* owner, const char* repo, const char* title, const char* body, const char* head, const char* base, const char* token);

// Utility functions
char* nl_github_get_token_from_env(void);
int64_t nl_github_check_rate_limit(const char* token);

#endif // NANOLANG_GITHUB_H
