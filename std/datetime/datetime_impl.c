/*
 * Datetime Implementation for Nanolang
 * Wraps standard C time.h functionality
 */

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

typedef struct {
    struct tm tm;
    time_t timestamp;
    int is_valid;
} nl_datetime_t;

// --- Constructors ---

void* nl_datetime_now(void) {
    nl_datetime_t* dt = (nl_datetime_t*)malloc(sizeof(nl_datetime_t));
    if (!dt) return NULL;
    
    dt->timestamp = time(NULL);
    struct tm* local = localtime(&dt->timestamp);
    if (local) {
        dt->tm = *local;
        dt->is_valid = 1;
    } else {
        dt->is_valid = 0;
    }
    
    return dt;
}

void* nl_datetime_parse_iso(const char* datetime_str) {
    if (!datetime_str) return NULL;
    
    nl_datetime_t* dt = (nl_datetime_t*)malloc(sizeof(nl_datetime_t));
    if (!dt) return NULL;
    
    memset(&dt->tm, 0, sizeof(struct tm));
    
    // Try full ISO format: YYYY-MM-DDTHH:MM:SS
    if (strptime(datetime_str, "%Y-%m-%dT%H:%M:%S", &dt->tm) != NULL) {
        dt->timestamp = mktime(&dt->tm);
        dt->is_valid = 1;
        return dt;
    }
    
    // Try date only: YYYY-MM-DD
    if (strptime(datetime_str, "%Y-%m-%d", &dt->tm) != NULL) {
        dt->timestamp = mktime(&dt->tm);
        dt->is_valid = 1;
        return dt;
    }
    
    free(dt);
    return NULL;
}

void* nl_datetime_parse_format(const char* datetime_str, const char* format) {
    if (!datetime_str || !format) return NULL;
    
    nl_datetime_t* dt = (nl_datetime_t*)malloc(sizeof(nl_datetime_t));
    if (!dt) return NULL;
    
    memset(&dt->tm, 0, sizeof(struct tm));
    
    if (strptime(datetime_str, format, &dt->tm) != NULL) {
        dt->timestamp = mktime(&dt->tm);
        dt->is_valid = 1;
        return dt;
    }
    
    free(dt);
    return NULL;
}

void* nl_datetime_create(int64_t year, int64_t month, int64_t day,
                         int64_t hour, int64_t minute, int64_t second) {
    nl_datetime_t* dt = (nl_datetime_t*)malloc(sizeof(nl_datetime_t));
    if (!dt) return NULL;
    
    memset(&dt->tm, 0, sizeof(struct tm));
    dt->tm.tm_year = year - 1900;
    dt->tm.tm_mon = month - 1;
    dt->tm.tm_mday = day;
    dt->tm.tm_hour = hour;
    dt->tm.tm_min = minute;
    dt->tm.tm_sec = second;
    dt->tm.tm_isdst = -1;  // Auto-detect DST
    
    dt->timestamp = mktime(&dt->tm);
    dt->is_valid = 1;
    
    return dt;
}

void* nl_datetime_from_timestamp(int64_t timestamp) {
    nl_datetime_t* dt = (nl_datetime_t*)malloc(sizeof(nl_datetime_t));
    if (!dt) return NULL;
    
    dt->timestamp = (time_t)timestamp;
    struct tm* local = localtime(&dt->timestamp);
    if (local) {
        dt->tm = *local;
        dt->is_valid = 1;
    } else {
        dt->is_valid = 0;
    }
    
    return dt;
}

// --- Formatters ---

const char* nl_datetime_to_iso(void* datetime) {
    if (!datetime) return NULL;
    
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    if (!dt->is_valid) return NULL;
    
    char* buffer = (char*)malloc(32);
    if (!buffer) return NULL;
    
    strftime(buffer, 32, "%Y-%m-%dT%H:%M:%S", &dt->tm);
    return buffer;
}

const char* nl_datetime_format(void* datetime, const char* format) {
    if (!datetime || !format) return NULL;
    
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    if (!dt->is_valid) return NULL;
    
    char* buffer = (char*)malloc(256);
    if (!buffer) return NULL;
    
    strftime(buffer, 256, format, &dt->tm);
    return buffer;
}

int64_t nl_datetime_to_timestamp(void* datetime) {
    if (!datetime) return 0;
    
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    if (!dt->is_valid) return 0;
    
    return (int64_t)dt->timestamp;
}

// --- Component Accessors ---

int64_t nl_datetime_year(void* datetime) {
    if (!datetime) return 0;
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    return dt->is_valid ? dt->tm.tm_year + 1900 : 0;
}

int64_t nl_datetime_month(void* datetime) {
    if (!datetime) return 0;
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    return dt->is_valid ? dt->tm.tm_mon + 1 : 0;
}

int64_t nl_datetime_day(void* datetime) {
    if (!datetime) return 0;
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    return dt->is_valid ? dt->tm.tm_mday : 0;
}

int64_t nl_datetime_hour(void* datetime) {
    if (!datetime) return 0;
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    return dt->is_valid ? dt->tm.tm_hour : 0;
}

int64_t nl_datetime_minute(void* datetime) {
    if (!datetime) return 0;
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    return dt->is_valid ? dt->tm.tm_min : 0;
}

int64_t nl_datetime_second(void* datetime) {
    if (!datetime) return 0;
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    return dt->is_valid ? dt->tm.tm_sec : 0;
}

int64_t nl_datetime_weekday(void* datetime) {
    if (!datetime) return 0;
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    return dt->is_valid ? dt->tm.tm_wday : 0;
}

int64_t nl_datetime_day_of_year(void* datetime) {
    if (!datetime) return 0;
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    return dt->is_valid ? dt->tm.tm_yday + 1 : 0;
}

// --- Arithmetic ---

void* nl_datetime_add_seconds(void* datetime, int64_t seconds) {
    if (!datetime) return NULL;
    
    nl_datetime_t* dt = (nl_datetime_t*)datetime;
    if (!dt->is_valid) return NULL;
    
    return nl_datetime_from_timestamp(dt->timestamp + seconds);
}

void* nl_datetime_add_minutes(void* datetime, int64_t minutes) {
    return nl_datetime_add_seconds(datetime, minutes * 60);
}

void* nl_datetime_add_hours(void* datetime, int64_t hours) {
    return nl_datetime_add_seconds(datetime, hours * 3600);
}

void* nl_datetime_add_days(void* datetime, int64_t days) {
    return nl_datetime_add_seconds(datetime, days * 86400);
}

int64_t nl_datetime_diff_seconds(void* dt1, void* dt2) {
    if (!dt1 || !dt2) return 0;
    
    nl_datetime_t* datetime1 = (nl_datetime_t*)dt1;
    nl_datetime_t* datetime2 = (nl_datetime_t*)dt2;
    
    if (!datetime1->is_valid || !datetime2->is_valid) return 0;
    
    return (int64_t)difftime(datetime1->timestamp, datetime2->timestamp);
}

// --- Comparison ---

int64_t nl_datetime_equals(void* dt1, void* dt2) {
    if (!dt1 || !dt2) return 0;
    
    nl_datetime_t* datetime1 = (nl_datetime_t*)dt1;
    nl_datetime_t* datetime2 = (nl_datetime_t*)dt2;
    
    if (!datetime1->is_valid || !datetime2->is_valid) return 0;
    
    return datetime1->timestamp == datetime2->timestamp ? 1 : 0;
}

int64_t nl_datetime_before(void* dt1, void* dt2) {
    if (!dt1 || !dt2) return 0;
    
    nl_datetime_t* datetime1 = (nl_datetime_t*)dt1;
    nl_datetime_t* datetime2 = (nl_datetime_t*)dt2;
    
    if (!datetime1->is_valid || !datetime2->is_valid) return 0;
    
    return datetime1->timestamp < datetime2->timestamp ? 1 : 0;
}

int64_t nl_datetime_after(void* dt1, void* dt2) {
    if (!dt1 || !dt2) return 0;
    
    nl_datetime_t* datetime1 = (nl_datetime_t*)dt1;
    nl_datetime_t* datetime2 = (nl_datetime_t*)dt2;
    
    if (!datetime1->is_valid || !datetime2->is_valid) return 0;
    
    return datetime1->timestamp > datetime2->timestamp ? 1 : 0;
}

// --- Utilities ---

int64_t nl_datetime_is_leap_year(int64_t year) {
    return ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)) ? 1 : 0;
}

int64_t nl_datetime_days_in_month(int64_t year, int64_t month) {
    int days_per_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    
    if (month < 1 || month > 12) return 0;
    
    int days = days_per_month[month - 1];
    
    // February in leap year
    if (month == 2 && nl_datetime_is_leap_year(year)) {
        days = 29;
    }
    
    return days;
}

void nl_datetime_free(void* datetime) {
    if (datetime) {
        free(datetime);
    }
}

