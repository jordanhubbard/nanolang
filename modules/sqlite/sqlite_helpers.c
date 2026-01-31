/**
 * sqlite_helpers.c - Simplified SQLite3 wrapper for nanolang
 * 
 * Provides database operations for local data storage.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sqlite3.h>

/**
 * Get SQLite version string
 */
const char* nl_sqlite3_version(void) {
    return sqlite3_libversion();
}

/**
 * Get SQLite version number
 */
int64_t nl_sqlite3_version_number(void) {
    return (int64_t)sqlite3_libversion_number();
}

/**
 * Open a database
 * Returns database handle (pointer as int64) or 0 on failure
 */
int64_t nl_sqlite3_open(const char *filename) {
    sqlite3 *db;
    int rc = sqlite3_open(filename, &db);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return 0;
    }
    
    return (int64_t)db;
}

/**
 * Close a database
 * Returns 0 on success
 */
int64_t nl_sqlite3_close(int64_t db_handle) {
    sqlite3 *db = (sqlite3 *)db_handle;
    if (!db) return 1;
    
    int rc = sqlite3_close(db);
    return (int64_t)rc;
}

/**
 * Execute SQL without returning results (CREATE, INSERT, UPDATE, DELETE)
 * Returns 0 on success, error code otherwise
 */
int64_t nl_sqlite3_exec(int64_t db_handle, const char *sql) {
    sqlite3 *db = (sqlite3 *)db_handle;
    if (!db) return 1;
    
    char *err_msg = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err_msg);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
    }
    
    return (int64_t)rc;
}

/**
 * Prepare a SQL statement
 * Returns statement handle or 0 on failure
 */
int64_t nl_sqlite3_prepare(int64_t db_handle, const char *sql) {
    sqlite3 *db = (sqlite3 *)db_handle;
    if (!db) return 0;
    
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    
    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        return 0;
    }
    
    return (int64_t)stmt;
}

/**
 * Finalize (free) a prepared statement
 * Returns 0 on success
 */
int64_t nl_sqlite3_finalize(int64_t stmt_handle) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 1;
    
    return (int64_t)sqlite3_finalize(stmt);
}

/**
 * Execute a prepared statement (step to next row)
 * Returns: 100 (SQLITE_ROW) if row available, 101 (SQLITE_DONE) if finished, error code otherwise
 */
int64_t nl_sqlite3_step(int64_t stmt_handle) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 1;
    
    return (int64_t)sqlite3_step(stmt);
}

/**
 * Reset a prepared statement to execute again
 * Returns 0 on success
 */
int64_t nl_sqlite3_reset(int64_t stmt_handle) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 1;
    
    return (int64_t)sqlite3_reset(stmt);
}

/**
 * Bind integer parameter to prepared statement
 * index: 1-based parameter index
 * Returns 0 on success
 */
int64_t nl_sqlite3_bind_int(int64_t stmt_handle, int64_t index, int64_t value) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 1;
    
    return (int64_t)sqlite3_bind_int64(stmt, (int)index, value);
}

/**
 * Bind float parameter to prepared statement
 * Returns 0 on success
 */
int64_t nl_sqlite3_bind_double(int64_t stmt_handle, int64_t index, double value) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 1;
    
    return (int64_t)sqlite3_bind_double(stmt, (int)index, value);
}

/**
 * Bind string parameter to prepared statement
 * Returns 0 on success
 */
int64_t nl_sqlite3_bind_text(int64_t stmt_handle, int64_t index, const char *value) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 1;
    
    return (int64_t)sqlite3_bind_text(stmt, (int)index, value, -1, SQLITE_TRANSIENT);
}

/**
 * Bind NULL parameter to prepared statement
 * Returns 0 on success
 */
int64_t nl_sqlite3_bind_null(int64_t stmt_handle, int64_t index) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 1;
    
    return (int64_t)sqlite3_bind_null(stmt, (int)index);
}

/**
 * Get number of columns in result set
 */
int64_t nl_sqlite3_column_count(int64_t stmt_handle) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 0;
    
    return (int64_t)sqlite3_column_count(stmt);
}

/**
 * Get column name by index (0-based)
 */
const char* nl_sqlite3_column_name(int64_t stmt_handle, int64_t index) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return "";
    
    const char *name = sqlite3_column_name(stmt, (int)index);
    return name ? name : "";
}

/**
 * Get integer value from column (0-based index)
 */
int64_t nl_sqlite3_column_int(int64_t stmt_handle, int64_t index) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 0;
    
    return (int64_t)sqlite3_column_int64(stmt, (int)index);
}

/**
 * Get float value from column
 */
double nl_sqlite3_column_double(int64_t stmt_handle, int64_t index) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 0.0;
    
    return sqlite3_column_double(stmt, (int)index);
}

/**
 * Get string value from column
 */
const char* nl_sqlite3_column_text(int64_t stmt_handle, int64_t index) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return "";
    
    const unsigned char *text = sqlite3_column_text(stmt, (int)index);
    return text ? (const char *)text : "";
}

/**
 * Get column type (1=INTEGER, 2=FLOAT, 3=TEXT, 4=BLOB, 5=NULL)
 */
int64_t nl_sqlite3_column_type(int64_t stmt_handle, int64_t index) {
    sqlite3_stmt *stmt = (sqlite3_stmt *)stmt_handle;
    if (!stmt) return 0;
    
    return (int64_t)sqlite3_column_type(stmt, (int)index);
}

/**
 * Get last insert row ID
 */
int64_t nl_sqlite3_last_insert_rowid(int64_t db_handle) {
    sqlite3 *db = (sqlite3 *)db_handle;
    if (!db) return 0;
    
    return (int64_t)sqlite3_last_insert_rowid(db);
}

/**
 * Get number of rows modified by last statement
 */
int64_t nl_sqlite3_changes(int64_t db_handle) {
    sqlite3 *db = (sqlite3 *)db_handle;
    if (!db) return 0;
    
    return (int64_t)sqlite3_changes(db);
}

/**
 * Get last error message
 */
const char* nl_sqlite3_errmsg(int64_t db_handle) {
    sqlite3 *db = (sqlite3 *)db_handle;
    if (!db) return "Invalid database handle";
    
    return sqlite3_errmsg(db);
}

/**
 * Begin transaction
 */
int64_t nl_sqlite3_begin_transaction(int64_t db_handle) {
    return nl_sqlite3_exec(db_handle, "BEGIN TRANSACTION");
}

/**
 * Commit transaction
 */
int64_t nl_sqlite3_commit(int64_t db_handle) {
    return nl_sqlite3_exec(db_handle, "COMMIT");
}

/**
 * Rollback transaction
 */
int64_t nl_sqlite3_rollback(int64_t db_handle) {
    return nl_sqlite3_exec(db_handle, "ROLLBACK");
}
